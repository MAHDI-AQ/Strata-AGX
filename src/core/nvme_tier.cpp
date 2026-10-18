// src/core/nvme_tier.cpp - DirectStorage L3 NVMe Storage Tier for RadixTree
#include "strata/core/nvme_tier.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace strata::core {

namespace fs = std::filesystem;

std::string NVMeStorageTier::detect_default_storage_path() {
    const char* env_path = std::getenv("STRATA_NVME_TIER_DIR");
    if (env_path && std::strlen(env_path) > 0) {
        return std::string(env_path);
    }
    std::error_code ec;
    const char* home = std::getenv("HOME");
    if (home) {
        std::string home_path = std::string(home) + "/.cache/strata/nvme_radix";
        fs::create_directories(home_path, ec);
        if (!ec) return home_path;
    }
    return "/tmp/strata_nvme_radix";
}

NVMeStorageTier::NVMeStorageTier(const std::string& base_path) {
    if (base_path.empty()) {
        base_path_ = detect_default_storage_path();
    } else {
        base_path_ = base_path;
    }
    init();
}

NVMeStorageTier::~NVMeStorageTier() {
    shutdown();
}

bool NVMeStorageTier::init() {
    std::error_code ec;
    fs::create_directories(base_path_, ec);
    if (ec) {
        std::fprintf(stderr, "strata nvme_tier: failed to create directory %s: %s\n",
                     base_path_.c_str(), ec.message().c_str());
        return false;
    }

    // Clean crash temporaries on startup
    clean_crash_temporaries();

    // Probe O_DIRECT support on the filesystem
    std::string test_file = base_path_ + "/.direct_probe.bin";
    int fd = ::open(test_file.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd >= 0) {
        AlignedBuffer buf(kDirectBlockAlignment);
        std::memset(buf.data(), 0xAB, kDirectBlockAlignment);
        ssize_t w = ::write(fd, buf.data(), kDirectBlockAlignment);
        if (w == static_cast<ssize_t>(kDirectBlockAlignment)) {
            direct_io_supported_ = true;
        } else {
            direct_io_supported_ = false;
        }
        ::close(fd);
        ::unlink(test_file.c_str());
    } else {
        direct_io_supported_ = false;
    }

    return true;
}

void NVMeStorageTier::shutdown() {
    // Retain directory on graceful shutdown
}

std::string NVMeStorageTier::node_file_path(int64_t node_id) const {
    return base_path_ + "/node_" + std::to_string(node_id) + ".bin";
}

std::string NVMeStorageTier::tmp_node_file_path(int64_t node_id) const {
    return base_path_ + "/.tmp_node_" + std::to_string(node_id) + "_" + std::to_string(::getpid()) + ".bin";
}

std::string NVMeStorageTier::manifest_file_path() const {
    return base_path_ + "/radix_manifest.bin";
}

std::string NVMeStorageTier::tmp_manifest_file_path() const {
    return base_path_ + "/.tmp_radix_manifest_" + std::to_string(::getpid()) + ".bin";
}

size_t NVMeStorageTier::clean_crash_temporaries() {
    size_t swept = 0;
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) return 0;

    for (const auto& entry : fs::directory_iterator(base_path_, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind(".tmp_node_", 0) == 0 || fname.rfind(".tmp.", 0) == 0 || fname.rfind(".tmp_", 0) == 0) {
            std::error_code rm_ec;
            if (fs::remove(entry.path(), rm_ec)) {
                swept++;
            }
        }
    }
    return swept;
}

size_t NVMeStorageTier::clean_orphans(const std::unordered_set<int64_t>& valid_node_ids) {
    size_t unlinked = 0;
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) return 0;

    for (const auto& entry : fs::directory_iterator(base_path_, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind("node_", 0) == 0 && fname.rfind(".bin") == fname.size() - 4) {
            std::string id_str = fname.substr(5, fname.size() - 9);
            try {
                int64_t nid = std::stoll(id_str);
                if (valid_node_ids.find(nid) == valid_node_ids.end()) {
                    std::error_code rm_ec;
                    if (fs::remove(entry.path(), rm_ec)) {
                        unlinked++;
                    }
                }
            } catch (...) {
                // Ignore parse errors on non-standard filenames
            }
        }
    }
    return unlinked;
}

bool NVMeStorageTier::write_node_direct(int64_t node_id, const void* data, size_t bytes) {
    if (!data || bytes == 0) return false;

    std::string tmp_path = tmp_node_file_path(node_id);
    std::string final_path = node_file_path(node_id);

    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (direct_io_supported_) {
        flags |= O_DIRECT;
    }

    int fd = ::open(tmp_path.c_str(), flags, 0644);
    if (fd < 0) {
        // Fallback without O_DIRECT if flag caused error
        if (direct_io_supported_) {
            fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        }
        if (fd < 0) return false;
    }
#if defined(__linux__)
    // Pre-allocate blocks via fallocate to prevent fragmentation during Direct I/O
    ::fallocate(fd, 0, 0, static_cast<off_t>(bytes));
#endif

    size_t write_bytes = bytes;
    const void* write_ptr = data;
    AlignedBuffer staging;

    if (direct_io_supported_) {
        size_t aligned_len = AlignedBuffer::align_up(bytes, kDirectBlockAlignment);
        if ((reinterpret_cast<uintptr_t>(data) % kDirectBlockAlignment != 0) || (bytes != aligned_len)) {
            staging.allocate(aligned_len);
            if (staging.data() && bytes > 0) {
                std::memcpy(staging.data(), data, bytes);
            }
            write_ptr = staging.data();
            write_bytes = aligned_len;
        }
    }

    size_t total_written = 0;
    const uint8_t* p = static_cast<const uint8_t*>(write_ptr);
    bool write_ok = true;
    while (total_written < write_bytes) {
        ssize_t rc = ::write(fd, p + total_written, write_bytes - total_written);
        if (rc <= 0) {
            if (rc < 0 && errno == EINTR) continue;
            write_ok = false;
            break;
        }
        total_written += rc;
    }

    // Force commit to disk to eliminate volatile NAND cache drops
    if (write_ok) {
        if (::fdatasync(fd) != 0) {
            write_ok = false;
        }
    }

    ::close(fd);

    if (!write_ok) {
        ::unlink(tmp_path.c_str());
        return false;
    }

    // Two-Phase Atomic Rename: instantly replaces or creates target file with 0 partial window
    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        return false;
    }

    offloaded_nodes_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(write_bytes, std::memory_order_relaxed);
    return true;
}

bool NVMeStorageTier::read_node_direct(int64_t node_id, void* out_data, size_t bytes) {
    if (!out_data || bytes == 0) return false;

    std::string path = node_file_path(node_id);
    int flags = O_RDONLY;
    if (direct_io_supported_) {
        flags |= O_DIRECT;
    }

    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        if (direct_io_supported_) {
            fd = ::open(path.c_str(), O_RDONLY);
        }
        if (fd < 0) return false;
    }

    size_t read_bytes = bytes;
    void* read_ptr = out_data;
    AlignedBuffer staging;

    if (direct_io_supported_) {
        size_t aligned_len = AlignedBuffer::align_up(bytes, kDirectBlockAlignment);
        if ((reinterpret_cast<uintptr_t>(out_data) % kDirectBlockAlignment != 0) || (bytes != aligned_len)) {
            staging.allocate(aligned_len);
            read_ptr = staging.data();
            read_bytes = aligned_len;
        }
    }

    size_t total_read = 0;
    uint8_t* p = static_cast<uint8_t*>(read_ptr);
    while (total_read < read_bytes) {
        ssize_t rc = ::read(fd, p + total_read, read_bytes - total_read);
        if (rc < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return false;
        }
        if (rc == 0) break; // EOF
        total_read += rc;
    }

    ::close(fd);

    if (read_ptr != out_data) {
        std::memcpy(out_data, staging.data(), bytes);
    }

    bytes_read_.fetch_add(total_read, std::memory_order_relaxed);
    return total_read >= bytes;
}

bool NVMeStorageTier::has_node(int64_t node_id) const {
    std::string path = node_file_path(node_id);
    return ::access(path.c_str(), F_OK) == 0;
}

bool NVMeStorageTier::remove_node(int64_t node_id) {
    std::string path = node_file_path(node_id);
    if (::unlink(path.c_str()) == 0) {
        size_t prev = offloaded_nodes_.load(std::memory_order_relaxed);
        if (prev > 0) offloaded_nodes_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void NVMeStorageTier::purge() {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(base_path_, ec)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name.rfind("node_", 0) == 0 && name.rfind(".bin") == name.size() - 4) {
                fs::remove(entry.path(), ec);
            }
        }
    }
    offloaded_nodes_.store(0, std::memory_order_relaxed);
}

bool NVMeStorageTier::write_manifest(const std::string& model_sha256, const std::vector<NVMeManifestNodeRecord>& records) {
    std::string tmp_manifest = tmp_manifest_file_path();
    std::string final_manifest = manifest_file_path();

    int fd = ::open(tmp_manifest.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    NVMeManifestHeader hdr;
    hdr.magic = kNVMeMagic;
    hdr.version = kNVMeVersion;
    hdr.num_nodes = static_cast<uint32_t>(records.size());
    hdr.max_node_id = 0;
    std::memset(hdr.model_id_hash, 0, sizeof(hdr.model_id_hash));
    std::strncpy(hdr.model_id_hash, model_sha256.c_str(), sizeof(hdr.model_id_hash) - 1);

    for (const auto& r : records) {
        if (static_cast<uint64_t>(r.node_id) > hdr.max_node_id) {
            hdr.max_node_id = static_cast<uint64_t>(r.node_id);
        }
    }

    if (::write(fd, &hdr, sizeof(NVMeManifestHeader)) != sizeof(NVMeManifestHeader)) {
        ::close(fd);
        ::unlink(tmp_manifest.c_str());
        return false;
    }

    for (const auto& r : records) {
        if (::write(fd, &r, sizeof(NVMeManifestNodeRecord)) != sizeof(NVMeManifestNodeRecord)) {
            ::close(fd);
            ::unlink(tmp_manifest.c_str());
            return false;
        }
    }

    if (::fdatasync(fd) != 0) {
        ::close(fd);
        ::unlink(tmp_manifest.c_str());
        return false;
    }

    ::close(fd);
    return (::rename(tmp_manifest.c_str(), final_manifest.c_str()) == 0);
}

bool NVMeStorageTier::load_manifest(std::string& out_model_sha256, std::vector<NVMeManifestNodeRecord>& out_records) {
    std::string path = manifest_file_path();
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    NVMeManifestHeader hdr;
    if (::read(fd, &hdr, sizeof(NVMeManifestHeader)) != sizeof(NVMeManifestHeader)) {
        ::close(fd);
        return false;
    }

    if (hdr.magic != kNVMeMagic || hdr.version != kNVMeVersion) {
        ::close(fd);
        return false;
    }

    out_model_sha256 = std::string(hdr.model_id_hash, ::strnlen(hdr.model_id_hash, sizeof(hdr.model_id_hash)));
    out_records.clear();
    out_records.resize(hdr.num_nodes);

    for (uint32_t i = 0; i < hdr.num_nodes; ++i) {
        if (::read(fd, &out_records[i], sizeof(NVMeManifestNodeRecord)) != sizeof(NVMeManifestNodeRecord)) {
            ::close(fd);
            out_records.clear();
            return false;
        }
    }

    ::close(fd);
    return true;
}

double NVMeStorageTier::benchmark_write_throughput_mb_s(size_t total_bytes, size_t chunk_bytes) {
    chunk_bytes = AlignedBuffer::align_up(chunk_bytes, kDirectBlockAlignment);
    total_bytes = AlignedBuffer::align_up(total_bytes, chunk_bytes);

    AlignedBuffer buf(chunk_bytes);
    std::memset(buf.data(), 0x5A, chunk_bytes);

    std::string bench_file = base_path_ + "/.bench_write.bin";
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (direct_io_supported_) flags |= O_DIRECT;

    int fd = ::open(bench_file.c_str(), flags, 0644);
    if (fd < 0) return 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t written = 0;
    while (written < total_bytes) {
        ssize_t rc = ::write(fd, buf.data(), chunk_bytes);
        if (rc <= 0) break;
        written += rc;
    }
    ::close(fd);
    auto t1 = std::chrono::high_resolution_clock::now();
    ::unlink(bench_file.c_str());

    double sec = std::chrono::duration<double>(t1 - t0).count();
    if (sec <= 0.0) return 0.0;
    return (static_cast<double>(written) / (1024.0 * 1024.0)) / sec;
}

double NVMeStorageTier::benchmark_read_throughput_mb_s(size_t total_bytes, size_t chunk_bytes) {
    chunk_bytes = AlignedBuffer::align_up(chunk_bytes, kDirectBlockAlignment);
    total_bytes = AlignedBuffer::align_up(total_bytes, chunk_bytes);

    AlignedBuffer wbuf(chunk_bytes);
    std::memset(wbuf.data(), 0x3C, chunk_bytes);

    std::string bench_file = base_path_ + "/.bench_read.bin";
    int wfd = ::open(bench_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | (direct_io_supported_ ? O_DIRECT : 0), 0644);
    if (wfd < 0) return 0.0;
    size_t prep = 0;
    while (prep < total_bytes) {
        ssize_t rc = ::write(wfd, wbuf.data(), chunk_bytes);
        if (rc <= 0) break;
        prep += rc;
    }
    ::close(wfd);

    AlignedBuffer rbuf(chunk_bytes);
    int rfd = ::open(bench_file.c_str(), O_RDONLY | (direct_io_supported_ ? O_DIRECT : 0));
    if (rfd < 0) {
        ::unlink(bench_file.c_str());
        return 0.0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t nread = 0;
    while (nread < total_bytes) {
        ssize_t rc = ::read(rfd, rbuf.data(), chunk_bytes);
        if (rc <= 0) break;
        nread += rc;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    ::close(rfd);
    ::unlink(bench_file.c_str());

    double sec = std::chrono::duration<double>(t1 - t0).count();
    if (sec <= 0.0) return 0.0;
    return (static_cast<double>(nread) / (1024.0 * 1024.0)) / sec;
}

}  // namespace strata::core
