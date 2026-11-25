// src/core/boot_weight_crc.cu - Phase B0: opt-in 4-GPU boot VRAM weight CRC verification.
#include "strata/core/boot_weight_crc.hpp"
#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {
namespace {

struct BootCrcDevDesc {
    const uint8_t* ptr;
    uint64_t bytes;
};

__device__ __forceinline__ uint64_t rotl64_dev(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t splitmix64_mix_dev(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void boot_crc_verify_kernel(const BootCrcDevDesc* __restrict__ descs, uint64_t* __restrict__ out) {
    __shared__ uint64_t s[256];
    const int tid = (int) threadIdx.x;
    const BootCrcDevDesc d = descs[blockIdx.x];
    const uint8_t* p = d.ptr;
    const uint64_t bytes = d.bytes;

    uint64_t acc = 0x9e3779b97f4a7c15ULL ^ (uint64_t) tid ^ (bytes * 0xbf58476d1ce4e5b9ULL);
    const uint64_t full_words = bytes / 8;
    if ((((uintptr_t) p) & 7u) == 0) {
        const uint64_t* pw = reinterpret_cast<const uint64_t*>(p);
        for (uint64_t idx = (uint64_t) tid; idx < full_words; idx += 256) {
            acc = splitmix64_mix_dev(acc ^ pw[idx]);
        }
    } else {
        for (uint64_t idx = (uint64_t) tid; idx < full_words; idx += 256) {
            const uint8_t* bp = p + idx * 8;
            uint64_t w = 0;
            for (int b = 0; b < 8; ++b) w |= ((uint64_t) bp[b]) << (8 * b);
            acc = splitmix64_mix_dev(acc ^ w);
        }
    }

    const unsigned n_tail = (unsigned) (bytes & 7u);
    if (tid == 0 && n_tail != 0) {
        const uint8_t* tp = p + full_words * 8;
        uint64_t tail_word = 0;
        for (unsigned b = 0; b < n_tail; ++b) {
            tail_word |= ((uint64_t) tp[b]) << (8u * b);
        }
        acc = splitmix64_mix_dev(acc ^ (tail_word | ((uint64_t) n_tail << 56)));
    }

    s[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] = splitmix64_mix_dev(s[tid] ^ rotl64_dev(s[tid + stride], 17));
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

BootCrcTable& global_table() {
    static BootCrcTable tbl;
    return tbl;
}

}  // namespace

bool boot_weight_crc_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_BOOT_WEIGHT_CRC");
        return v != nullptr && v[0] != '0' && v[0] != '\0';
    }();
    return on;
}

void boot_crc_record(int dev, const char* kind, const char* tag, const void* dev_ptr, uint64_t bytes, const void* host_ptr) {
    global_table().record(dev, kind, tag, dev_ptr, bytes, host_ptr);
}

void boot_crc_clear_holder(int dev, const char* kind) {
    global_table().clear_holder(dev, kind);
}

void boot_crc_clear_all() {
    global_table().clear_all();
}

size_t boot_crc_region_count() {
    return global_table().size();
}

bool boot_weight_crc_verify_all(std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<BootCrcEntry> entries = global_table().snapshot();
    if (entries.empty()) {
        std::fprintf(stderr, "strata generate: STRATA_BOOT_WEIGHT_CRC=1 PASS 0/0 GPUs 0.00 GiB (0 slices, 0.0 ms) verified\n");
        return true;
    }

    std::map<int, std::vector<size_t>> by_dev;
    uint64_t total_bytes = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        by_dev[entries[i].dev].push_back(i);
        total_bytes += entries[i].bytes;
    }

    struct DevResult {
        int dev = -1;
        bool ok = true;
        std::string err;
    };
    std::vector<DevResult> results(by_dev.size());
    std::vector<std::thread> threads;
    threads.reserve(by_dev.size());

    size_t slot_i = 0;
    for (const auto& kv : by_dev) {
        const int dev = kv.first;
        const std::vector<size_t>* idxs_ptr = &kv.second;
        DevResult* res_ptr = &results[slot_i++];
        res_ptr->dev = dev;
        threads.emplace_back([&entries, dev, idxs_ptr, res_ptr]() {
            const std::vector<size_t>& idxs = *idxs_ptr;
            DevResult& res = *res_ptr;
            if (dev < 0) {
                res.ok = false;
                res.err = "boot_weight_crc: invalid CUDA device index (" + std::to_string(dev) + ")";
                return;
            }
            const OnDevice on_dev(dev);
            int cur_dev = -1;
            if (cudaGetDevice(&cur_dev) != cudaSuccess || cur_dev != dev) {
                (void) cudaGetLastError();
                res.ok = false;
                res.err = "boot_weight_crc: cannot select CUDA" + std::to_string(dev);
                return;
            }

            const size_t n = idxs.size();
            std::vector<BootCrcDevDesc> h_descs(n);
            std::vector<uint64_t> h_got(n, 0);
            for (size_t k = 0; k < n; ++k) {
                const BootCrcEntry& e = entries[idxs[k]];
                h_descs[k].ptr = static_cast<const uint8_t*>(e.dev_ptr);
                h_descs[k].bytes = e.bytes;
            }

            cudaStream_t stream = nullptr;
            void* dev_buf = nullptr;
            auto cleanup = [&]() {
                if (dev_buf) (void) cudaFree(dev_buf);
                if (stream) (void) cudaStreamDestroy(stream);
            };

            cudaError_t st = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (st != cudaSuccess) {
                (void) cudaGetLastError();
                res.ok = false;
                res.err = std::string("boot_weight_crc: CUDA") + std::to_string(dev) + " stream create: " + cudaGetErrorString(st);
                cleanup();
                return;
            }

            const size_t desc_bytes = n * sizeof(BootCrcDevDesc);
            const size_t out_bytes = n * sizeof(uint64_t);
            st = cudaMalloc(&dev_buf, desc_bytes + out_bytes);
            if (st != cudaSuccess) {
                (void) cudaGetLastError();
                res.ok = false;
                res.err = std::string("boot_weight_crc: CUDA") + std::to_string(dev) + " alloc failed: " + cudaGetErrorString(st);
                cleanup();
                return;
            }
            auto* d_descs = static_cast<BootCrcDevDesc*>(dev_buf);
            auto* d_out = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(dev_buf) + desc_bytes);

            st = cudaMemcpyAsync(d_descs, h_descs.data(), desc_bytes, cudaMemcpyHostToDevice, stream);
            if (st == cudaSuccess) {
                boot_crc_verify_kernel<<<(unsigned) n, 256, 0, stream>>>(d_descs, d_out);
                st = cudaPeekAtLastError();
            }
            if (st == cudaSuccess) {
                st = cudaMemcpyAsync(h_got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, stream);
            }
            if (st == cudaSuccess) {
                st = cudaStreamSynchronize(stream);
            }
            cleanup();

            if (st != cudaSuccess) {
                (void) cudaGetLastError();
                res.ok = false;
                res.err = std::string("boot_weight_crc: CUDA") + std::to_string(dev) + " verify failed: " + cudaGetErrorString(st);
                return;
            }

            for (size_t k = 0; k < n; ++k) {
                const BootCrcEntry& e = entries[idxs[k]];
                if (h_got[k] != e.expected) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "(CUDA%d %s:%s bytes=%" PRIu64 " expected=0x%016" PRIx64 " got=0x%016" PRIx64 ")",
                                  e.dev, e.kind, e.tag, e.bytes, e.expected, h_got[k]);
                    if (res.ok) {
                        res.ok = false;
                        res.err = std::string("boot_weight_crc MISMATCH ") + buf;
                    }
                    std::fprintf(stderr, "strata generate: STRATA_BOOT_WEIGHT_CRC=1 FAIL %s\n", buf);
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    for (const DevResult& r : results) {
        if (!r.ok) {
            err = r.err;
            return false;
        }
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "strata generate: STRATA_BOOT_WEIGHT_CRC=1 PASS %zu/%zu GPUs %.2f GiB (%zu slices, %.1f ms) verified\n",
                 by_dev.size(), by_dev.size(), (double) total_bytes / 1073741824.0, entries.size(), elapsed_ms);
    return true;
}

}  // namespace strata::core
