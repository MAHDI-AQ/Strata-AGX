// src/core/nvme_tier.cpp - DirectStorage L3 NVMe Storage Tier for RadixTree
#include "strata/core/nvme_tier.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace strata::core {

std::string NVMeStorageTier::detect_default_storage_path() {
    const char* env_path = std::getenv("STRATA_NVME_TIER_DIR");
    if (env_path && std::strlen(env_path) > 0) {
        return std::string(env_path);
    }
    // Placement is explicit: STRATA_NVME_TIER_DIR (checked above) names the fast-disk location;
    // without it, the per-user cache is used.
    std::error_code ec;
    // User cache fallback
    const char* home = std::getenv("HOME");
    if (home) {
        std::string home_path = std::string(home) + "/.cache/strata/nvme_radix";
        fs::create_directories(home_path, ec);
        if (!ec) return home_path;
    }
    // Final temporary fallback
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
    // Retain state directory or cleanup on purge
}

std::string NVMeStorageTier::node_file_path(int64_t node_id) const {
    return base_path_ + "/node_" + std::to_string(node_id) + ".bin";
}

bool NVMeStorageTier::write_node_direct(int64_t node_id, const void* data, size_t bytes) {
    if (!data || bytes == 0) return false;

    std::string path = node_file_path(node_id);
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (direct_io_supported_) {
        flags |= O_DIRECT;
    }

    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
        // Fallback without O_DIRECT if flag caused error
        if (direct_io_supported_) {
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        }
        if (fd < 0) return false;
    }
#if defined(__linux__)
    // Pre-allocate blocks via native Linux fallocate to eliminate ext4 metadata journal locks during Direct I/O
    ::fallocate(fd, 0, 0, static_cast<off_t>(bytes));
#endif

    size_t write_bytes = bytes;
    const void* write_ptr = data;
    AlignedBuffer staging;

    if (direct_io_supported_) {
        size_t aligned_len = AlignedBuffer::align_up(bytes, kDirectBlockAlignment);
        // Check if data is already page-aligned
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
    while (total_written < write_bytes) {
        ssize_t rc = ::write(fd, p + total_written, write_bytes - total_written);
        if (rc <= 0) {
            ::close(fd);
            ::unlink(path.c_str()); // CRIT-05: erase corrupted partial file on failure
            return false;
        }
        if ((rc % kDirectBlockAlignment) != 0 && (total_written + rc) < write_bytes) {
            ::close(fd);
            ::unlink(path.c_str());
            return false;
        }
        total_written += rc;
    }

    ::close(fd);
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

    // Prepare fixture file
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
