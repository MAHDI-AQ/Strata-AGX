#pragma once
#include <cstdint>
#include <atomic>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace strata::ipc {

enum MsgType : uint8_t {
    MSG_NONE = 0,
    MSG_TOKEN = 1,
    MSG_PROGRESS = 2,
    MSG_DONE = 3,
    MSG_ERROR = 4
};

// Fixed 32-byte binary message struct
struct ShmMsg {
    std::atomic<uint8_t> type{MSG_NONE};
    uint8_t flags{0};
    uint16_t reserved{0};
    uint32_t token{0};
    uint64_t id{0};
    int64_t progress_read{0};
    int64_t progress_total{0};
};

static constexpr size_t kShmCapacity = 4096;

struct ShmRing {
    std::atomic<uint32_t> head{0};  // Producer (C++) writes here
    std::atomic<uint32_t> tail{0};  // Consumer (Python) reads here
    uint32_t capacity{kShmCapacity};
    uint32_t magic{0x53545241};     // 'STRA'
    ShmMsg entries[kShmCapacity];
};

class ShmProducer {
public:
    static ShmProducer* instance() {
        static ShmProducer inst;
        return &inst;
    }
    bool init(const std::string& name) {
        if (name.empty()) return false;
        name_ = name[0] == '/' ? name : ("/" + name);
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) return false;
        if (ftruncate(fd_, sizeof(ShmRing)) != 0) { close(fd_); fd_ = -1; return false; }
        void* ptr = mmap(nullptr, sizeof(ShmRing), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr == MAP_FAILED) { close(fd_); fd_ = -1; return false; }
        ring_ = static_cast<ShmRing*>(ptr);
        ring_->magic = 0x53545241;
        ring_->capacity = kShmCapacity;
        ring_->head.store(0, std::memory_order_relaxed);
        ring_->tail.store(0, std::memory_order_relaxed);
        active_ = true;
        return true;
    }
    bool is_active() const { return active_; }

    bool write_token(uint64_t id, int32_t token) {
        if (!active_ || !ring_) return false;
        uint32_t h = ring_->head.load(std::memory_order_relaxed);
        // Task 1.3: Bounded backpressure spin instead of silent drop
        int spins = 0;
        while (true) {
            uint32_t t = ring_->tail.load(std::memory_order_acquire);
            if (h - t < ring_->capacity) break;
            if (++spins > 200000) {
                // If consumer is dead/stalled for >20ms, drop rather than deadlocking engine
                return false;
            }
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#endif
        }
        uint32_t idx = h & (ring_->capacity - 1);
        auto& m = ring_->entries[idx];
        m.id = id;
        m.token = static_cast<uint32_t>(token);
        m.progress_read = 0;
        m.progress_total = 0;
        m.flags = 0;
        m.type.store(MSG_TOKEN, std::memory_order_release);
        ring_->head.store(h + 1, std::memory_order_release);
        return true;
    }

    bool write_progress(uint64_t id, int64_t read, int64_t total) {
        if (!active_ || !ring_) return false;
        uint32_t h = ring_->head.load(std::memory_order_relaxed);
        uint32_t t = ring_->tail.load(std::memory_order_acquire);
        if (h - t >= ring_->capacity) return false;
        uint32_t idx = h & (ring_->capacity - 1);
        auto& m = ring_->entries[idx];
        m.id = id;
        m.token = 0;
        m.progress_read = read;
        m.progress_total = total;
        m.flags = 0;
        m.type.store(MSG_PROGRESS, std::memory_order_release);
        ring_->head.store(h + 1, std::memory_order_release);
        return true;
    }

    ~ShmProducer() {
        if (ring_) { munmap(ring_, sizeof(ShmRing)); ring_ = nullptr; }
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
        // shm unlinked by python supervisor
    }
private:
    ShmProducer() = default;
    int fd_{-1};
    ShmRing* ring_{nullptr};
    std::string name_;
    bool active_{false};
};

} // namespace strata::ipc
