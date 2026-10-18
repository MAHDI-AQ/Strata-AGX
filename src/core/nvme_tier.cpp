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
#include <csignal>
#include <atomic>
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
        if (buf.data()) {
            std::memset(buf.data(), 0xAB, kDirectBlockAlignment);
            ssize_t w = ::write(fd, buf.data(), kDirectBlockAlignment);
            if (w == static_cast<ssize_t>(kDirectBlockAlignment)) {
                direct_io_supported_ = true;
            } else {
                direct_io_supported_ = false;
            }
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
    static std::atomic<uint64_t> s_tmp_seq{0};
    return base_path_ + "/.tmp_node_" + std::to_string(node_id) + "_" +
           std::to_string(::getpid()) + "_" +
           std::to_string(s_tmp_seq.fetch_add(1, std::memory_order_relaxed)) + ".bin";
}

std::string NVMeStorageTier::manifest_file_path() const {
    return base_path_ + "/radix_manifest.bin";
}

std::string NVMeStorageTier::tmp_manifest_file_path() const {
    static std::atomic<uint64_t> s_manifest_seq{0};
    return base_path_ + "/.tmp_radix_manifest_" + std::to_string(::getpid()) + "_" +
           std::to_string(s_manifest_seq.fetch_add(1, std::memory_order_relaxed)) + ".bin";
}

size_t NVMeStorageTier::clean_crash_temporaries() {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    size_t swept = 0;
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) return 0;

    const auto now = std::filesystem::file_time_type::clock::now();

    for (const auto& entry : fs::directory_iterator(base_path_, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind(".tmp_node_", 0) == 0 || fname.rfind(".tmp.", 0) == 0 || fname.rfind(".tmp_", 0) == 0) {
            bool safe_to_delete = false;
            // Attempt to parse PID from .tmp_node_<id>_<pid>...
            if (fname.rfind(".tmp_node_", 0) == 0) {
                size_t p1 = fname.find('_', 10); // after .tmp_node_
                if (p1 != std::string::npos) {
                    size_t p2 = fname.find('_', p1 + 1);
                    if (p2 == std::string::npos) {
                        p2 = fname.find(".bin", p1 + 1);
                    }
                    if (p2 != std::string::npos && p2 > p1 + 1) {
                        std::string pid_str = fname.substr(p1 + 1, p2 - (p1 + 1));
                        try {
                            pid_t pid = static_cast<pid_t>(std::stol(pid_str));
                            if (pid > 0) {
                                int res = ::kill(pid, 0);
                                if (res == -1 && errno == ESRCH) {
                                    safe_to_delete = true;
                                } else {
                                    // Process is active, skip
                                    continue;
                                }
                            }
                        } catch (...) {}
                    }
                }
            }
            if (!safe_to_delete) {
                // If PID couldn't be parsed, only delete if older than 300 seconds
                auto lwt = entry.last_write_time(ec);
                if (!ec) {
                    auto age_sec = std::chrono::duration_cast<std::chrono::seconds>(now - lwt).count();
                    if (age_sec > 300) {
                        safe_to_delete = true;
                    }
                }
            }
            if (safe_to_delete) {
                std::error_code rm_ec;
                if (fs::remove(entry.path(), rm_ec)) {
                    swept++;
                }
            }
        }
    }
    return swept;
}

size_t NVMeStorageTier::clean_orphans(const std::unordered_set<int64_t>& valid_node_ids) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    size_t unlinked = 0;
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) return 0;

    const auto now = std::filesystem::file_time_type::clock::now();

    for (const auto& entry : fs::directory_iterator(base_path_, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind("node_", 0) == 0 && fname.rfind(".bin") == fname.size() - 4) {
            std::string id_str = fname.substr(5, fname.size() - 9);
            try {
                int64_t nid = std::stoll(id_str);
                if (valid_node_ids.find(nid) == valid_node_ids.end()) {
                    // Grace Period: Skip any files modified within the last 60 seconds
                    auto lwt = entry.last_write_time(ec);
                    if (!ec) {
                        auto age_sec = std::chrono::duration_cast<std::chrono::seconds>(now - lwt).count();
                        if (age_sec < 60) {
                            continue;
                        }
                    }
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
    if (circuit_tripped_.load(std::memory_order_relaxed)) {
        return false;
    }
    if (!data || bytes == 0) return false;

    auto record_failure = [&](int err) {
        if (err == ENOSPC || consecutive_write_errors_.fetch_add(1, std::memory_order_relaxed) + 1 >= kMaxConsecutiveWriteErrors) {
            circuit_tripped_.store(true, std::memory_order_release);
            std::fprintf(stderr,
                "[NVMeStorageTier] CRITICAL: Circuit breaker TRIPPED (ENOSPC or write errors). Disabling L3 NVMe writes to protect engine stability.\n");
        }
    };

    std::shared_lock<std::shared_mutex> lk(mutex_);

    std::string tmp_path = tmp_node_file_path(node_id);
    std::string final_path = node_file_path(node_id);

    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (direct_io_supported_) {
        flags |= O_DIRECT;
    }

    int fd = ::open(tmp_path.c_str(), flags, 0644);
    if (fd < 0) {
        if (direct_io_supported_) {
            fd = ::open(tmp_path.c_str(), O_WRONLY| O_CREAT | O_TRUNC, 0644);
        }
        if (fd < 0) {
            record_failure(errno);
            return false;
        }
    }

    struct FileGuard {
        int fd;
        const std::string& path;
        bool committed = false;
        ~FileGuard() {
            if (fd >= 0) ::close(fd);
            if (!committed) {
                std::error_code ec;
                fs::remove(path, ec);
            }
        }
    } guard{fd, tmp_path};

    size_t write_bytes = bytes;
    const void* write_ptr = data;
    AlignedBuffer staging;

    if (direct_io_supported_) {
        size_t aligned_len = AlignedBuffer::align_up(bytes, kDirectBlockAlignment);
        if ((reinterpret_cast<uintptr_t>(data) % kDirectBlockAlignment != 0) || (bytes != aligned_len)) {
            staging.allocate(aligned_len);
            if (staging.empty() || !staging.data()) {
                return false;
            }
            std::memcpy(staging.data(), data, bytes);
            write_ptr = staging.data();
            write_bytes = aligned_len;
        }
    }

#if defined(__linux__)
    // Pre-allocate blocks via fallocate using actual sector-aligned Direct I/O size
    ::fallocate(fd, 0, 0, static_cast<off_t>(write_bytes));
#endif

    size_t total_written = 0;
    const uint8_t* p = static_cast<const uint8_t*>(write_ptr);
    while (total_written < write_bytes) {
        ssize_t rc = ::write(fd, p + total_written, write_bytes - total_written);
        if (rc <= 0) {
            if (rc < 0 && errno == EINTR) continue;
            record_failure(errno);
            return false;
        }
        total_written += rc;
    }

    // Force commit to disk to eliminate volatile NAND cache drops
    if (::fdatasync(fd) != 0) {
        record_failure(errno);
        return false;
    }

    ::close(guard.fd);
    guard.fd = -1;

    // Two-Phase Atomic Rename: replaces or creates target file with 0 partial window
    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        record_failure(errno);
        return false;
    }

    guard.committed = true;
    consecutive_write_errors_.store(0, std::memory_order_relaxed);
    offloaded_nodes_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(write_bytes, std::memory_order_relaxed);
    return true;
}

bool NVMeStorageTier::read_node_direct(int64_t node_id, void* out_data, size_t bytes) {
    if (!out_data || bytes == 0) return false;

    std::shared_lock<std::shared_mutex> lk(mutex_);

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
            if (staging.empty() || !staging.data()) {
                ::close(fd);
                return false;
            }
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

    if (total_read < bytes) {
        return false;
    }

    if (read_ptr != out_data) {
        std::memcpy(out_data, staging.data(), bytes);
    }

    // Hardware SSE4.2 CRC32C Validation & Auto-Unlink
    if (bytes >= sizeof(NVMeNodeHeader)) {
        const auto* hdr = reinterpret_cast<const NVMeNodeHeader*>(out_data);
        if (hdr->magic == kNVMeMagic && hdr->version == kNVMeVersion && hdr->checksum != 0) {
            uint64_t actual = compute_crc32c(
                static_cast<const uint8_t*>(out_data) + sizeof(NVMeNodeHeader),
                bytes - sizeof(NVMeNodeHeader));
            if (actual != hdr->checksum) {
                std::fprintf(stderr,
                    "[NVMeStorageTier] CORRUPTION DETECTED for node %ld: expected checksum %lu, computed %lu. Unlinking corrupted snapshot.\n",
                    static_cast<long>(node_id),
                    static_cast<unsigned long>(hdr->checksum),
                    static_cast<unsigned long>(actual));
                // CRIT-STORE-01: Prevent TOCTOU race with concurrent atomic rename by checking inode
                struct stat st_fd, st_path;
                if (::fstat(fd, &st_fd) == 0 && ::stat(path.c_str(), &st_path) == 0) {
                    if (st_fd.st_ino == st_path.st_ino) {
                        ::unlink(path.c_str());
                    }
                }
                return false;
            }
        }
    }

    bytes_read_.fetch_add(total_read, std::memory_order_relaxed);
    return true;
}

bool NVMeStorageTier::has_node(int64_t node_id) const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    std::string path = node_file_path(node_id);
    return ::access(path.c_str(), F_OK) == 0;
}

bool NVMeStorageTier::remove_node(int64_t node_id) {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    std::string path = node_file_path(node_id);
    if (::unlink(path.c_str()) == 0) {
        size_t prev = offloaded_nodes_.load(std::memory_order_relaxed);
        if (prev > 0) offloaded_nodes_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void NVMeStorageTier::purge() {
    std::unique_lock<std::shared_mutex> lk(mutex_);
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
    std::unique_lock<std::shared_mutex> lk(mutex_);
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
    std::shared_lock<std::shared_mutex> lk(mutex_);
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
    if (!buf.data()) return 0.0;
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
    if (!wbuf.data()) return 0.0;
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
    if (!rbuf.data()) {
        ::unlink(bench_file.c_str());
        return 0.0;
    }
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


// --- Phase 3: IOUringStorage DirectStorage Implementation ---
namespace strata::core {

IOUringStorage::IOUringStorage(size_t queue_depth) : queue_depth_(queue_depth) {
#if STRATA_HAS_IO_URING
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));
    ring_fd_ = syscall(__NR_io_uring_setup, static_cast<unsigned>(queue_depth_), &params);
    if (ring_fd_ >= 0) {
        uring_initialized_.store(true, std::memory_order_release);
    } else {
        uring_initialized_.store(false, std::memory_order_release);
    }
#else
    uring_initialized_.store(false, std::memory_order_release);
#endif
}

IOUringStorage::~IOUringStorage() {
    if (ring_fd_ >= 0) {
        ::close(ring_fd_);
        ring_fd_ = -1;
    }
}

int IOUringStorage::sync_direct_io(int fd, void* buf, size_t bytes, off_t offset, bool is_write) {
    size_t total = 0;
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    while (total < bytes) {
        ssize_t rc = is_write ? ::pwrite(fd, ptr + total, bytes - total, offset + total)
                              : ::pread(fd, ptr + total, bytes - total, offset + total);
        if (rc < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (rc == 0) break;
        total += rc;
    }
    return total == bytes ? 0 : -EIO;
}

int IOUringStorage::submit_read_direct(int fd, void* aligned_buf, size_t bytes, off_t offset) {
    return sync_direct_io(fd, aligned_buf, bytes, offset, false);
}

int IOUringStorage::submit_write_direct(int fd, const void* aligned_buf, size_t bytes, off_t offset) {
    return sync_direct_io(fd, const_cast<void*>(aligned_buf), bytes, offset, true);
}

int IOUringStorage::wait_completions(size_t) {
    return 0;
}

} // namespace strata::core
