// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

__device__ void resolve_one(KvStreamMap m, const int32_t* __restrict__ ids,
                               const int32_t* __restrict__ steps, int n_q, int cap, int page_size) {
    __shared__ int s_nmiss, s_lookups, s_cut, s_n_kv;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    if (threadIdx.x == 0) {
        s_nmiss = 0;
        s_lookups = 0;
        int max_n_kv = 0;
        for (int q = 0; q < n_q; ++q) {
            const int q_n_kv = steps[q * kStepCount + kStepNKv];
            if (q_n_kv > max_n_kv) max_n_kv = q_n_kv;
        }
        s_n_kv = max_n_kv;
    }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = threadIdx.x; i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                m.miss_block[atomicAdd(&s_nmiss, 1)] = b;
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // (#783, stuchapin909) most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    //    Lever 1: Attention Sink & Local Window Pinning Gate (old_b < 256 or local window >= n_kv - 32768)
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const int old_b = m.slot_block[j];
        const bool is_sink = (old_b >= 0 && old_b < 256);
        const bool is_local = (old_b >= 0 && s_n_kv > 32768 && old_b >= (s_n_kv - 32768) / page_size);
        const bool pinned = (old_b >= 0 && (is_sink || is_local));
        const bool pass2 = scanned >= 2 * n;
        const bool cand = !mine && (!pinned || pass2) && (old_b < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine && (!pinned || scanned >= n)) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap, int page_size) {
    resolve_one(m, ids, steps, n_q, cap, page_size);
}

// Lever 3: Hardware DMA / Coalesced Copy Optimization
// 1. Coalesces contiguous missed blocks (b, b+1, ...) into single contiguous multi-block transfers
// 2. Uses 4x uint4 vectorized unrolled loads/stores (64 bytes/thread) with 512B warp coalescence
//    to maximize PCIe queue depth and eliminate SM warp stall latency.
__global__ void copy_kernel(KvStreamMap m, Runs r) {
    const int need = m.ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        // If this miss is part of a contiguous run started earlier, skip it (already copied)
        if (k > 0 && m.miss_block[k - 1] + 1 == b && m.miss_slot[k - 1] + 1 == sl) continue;

        // Discover contiguous run length
        int run_len = 1;
        while (k + run_len < need &&
               m.miss_block[k + run_len] == b + run_len &&
               m.miss_slot[k + run_len] == sl + run_len) {
            ++run_len;
        }

        // Copy all runs coalesced
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            if (a >= r.n) break;
            const size_t total_u4 = (size_t) run_len * (r.len[a] / 16);
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            const size_t stride = (size_t) blockDim.x * 4;
            size_t base = 0;
            for (; base + stride <= total_u4; base += stride) {
                uint4 v0 = src[base + 0 * blockDim.x + threadIdx.x];
                uint4 v1 = src[base + 1 * blockDim.x + threadIdx.x];
                uint4 v2 = src[base + 2 * blockDim.x + threadIdx.x];
                uint4 v3 = src[base + 3 * blockDim.x + threadIdx.x];
                dst[base + 0 * blockDim.x + threadIdx.x] = v0;
                dst[base + 1 * blockDim.x + threadIdx.x] = v1;
                dst[base + 2 * blockDim.x + threadIdx.x] = v2;
                dst[base + 3 * blockDim.x + threadIdx.x] = v3;
            }
            for (size_t i = base + threadIdx.x; i < total_u4; i += blockDim.x) {
                dst[i] = src[i];
            }
        }
    }
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__global__ void invalidate_from_kernel(KvStreamMap m, long long min_block) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = min_block + i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        if (m.slot_block[i] >= (int32_t) min_block) {
            m.slot_block[i] = -1;
            m.slot_stamp[i] = -1;
            m.slot_ref[i] = 0;
        }
    }
}

__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) (i % n_slots);
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

bool kv_stream_invalidate_from(const KvStreamMap& m, int64_t min_block, void* stream) {
    if (min_block <= 0) {
        kv_stream_reset(m, stream);
        return cudaGetLastError() == cudaSuccess;
    }
    invalidate_from_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m, (long long) min_block);
    check("invalidate_from");
    return cudaGetLastError() == cudaSuccess;
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n", (long long) m.n_slots, RT);
        std::exit(1);
    }
    resolve_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(m, ids, steps, (int) n_q, (int) cap, (int) s.page_size);
    check("resolve");
    copy_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(m, runs_of(slots, host, fmt, s));
    check("copy");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

void kv_stage_block_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t block_idx,
                              const QsaShapes& s, void* stream) {
    if (block_idx < 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a] + block_idx * r.len[a], r.src[a] + block_idx * r.len[a],
                            (size_t) r.len[a], cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
            check("stage_block");
}

void kv_stage_selected_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt,
                                 const int32_t* sel_ids_dev, int64_t n_sel, int64_t n_prefix_blocks,
                                 const QsaShapes& s, void* stream) {
    if (n_prefix_blocks <= 0 || n_sel <= 0 || sel_ids_dev == nullptr) return;
    cudaStream_t cs = (cudaStream_t) stream;
    std::vector<int32_t> h_ids((size_t) n_sel);
    if (cudaMemcpyAsync(h_ids.data(), sel_ids_dev, (size_t) n_sel * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
        cudaStreamSynchronize(cs) != cudaSuccess) {
        check("stage_selected_ids");
        return;
    }
    std::vector<uint8_t> want((size_t) n_prefix_blocks, 0);
    for (int32_t b : h_ids) {
        if (b >= 0 && (int64_t) b < n_prefix_blocks) want[(size_t) b] = 1;
    }
    const Runs r = runs_of(stage, host, fmt, s);
    int64_t b = 0;
    while (b < n_prefix_blocks) {
        if (!want[(size_t) b]) { ++b; continue; }
        int64_t b_end = b + 1;
        while (b_end < n_prefix_blocks && want[(size_t) b_end]) ++b_end;
        const int64_t cnt = b_end - b;
        for (int a = 0; a < r.n; ++a) {
            if (cudaMemcpyAsync(r.dst[a] + b * r.len[a], r.src[a] + b * r.len[a],
                                (size_t) (cnt * r.len[a]), cudaMemcpyDefault, cs) != cudaSuccess)
                check("stage_selected");
        }
        b = b_end;
    }
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    if (b1 <= b0) return;
    const Runs r = runs_of(stage, host, fmt, s);   // src: the host copy, dst: the staging pool (identity layout both)
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync((void*) (r.src[a] + b0 * r.len[a]), r.dst[a] + b0 * r.len[a], (size_t) ((b1 - b0) * r.len[a]),
                            cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
            check("unstage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
