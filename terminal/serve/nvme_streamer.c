#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define ALIGNMENT (2 * 1024 * 1024)          // 2 MiB block alignment
#define DEFAULT_CHUNK_SIZE (4 * 1024 * 1024) // 4 MiB chunk
#define MAX_THREADS 32

typedef struct {
    int fd;
    uint8_t *buf;
    off_t offset;
    size_t size;
    size_t chunk_size;
    bool ring_mode;
} worker_arg_t;

static void *stream_read_worker(void *p) {
    worker_arg_t *arg = (worker_arg_t *)p;
    size_t remaining = arg->size;
    off_t cur = arg->offset;
    uint8_t *cur_buf = arg->buf;
    while (remaining > 0) {
        size_t to_read = (remaining < arg->chunk_size) ? remaining : arg->chunk_size;
        ssize_t n = pread(arg->fd, cur_buf, to_read, cur);
        if (n <= 0) break;
        remaining -= n;
        cur += n;
        if (!arg->ring_mode) {
            cur_buf += n;
        }
    }
    return NULL;
}

static void *stream_write_worker(void *p) {
    worker_arg_t *arg = (worker_arg_t *)p;
    size_t remaining = arg->size;
    off_t cur = arg->offset;
    const uint8_t *cur_buf = arg->buf;
    while (remaining > 0) {
        size_t to_write = (remaining < arg->chunk_size) ? remaining : arg->chunk_size;
        ssize_t n = pwrite(arg->fd, cur_buf, to_write, cur);
        if (n <= 0) break;
        remaining -= n;
        cur += n;
        if (!arg->ring_mode) {
            cur_buf += n;
        }
    }
    return NULL;
}

static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void *nvme_alloc_aligned(size_t size) {
    void *ptr = NULL;
    size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    if (posix_memalign(&ptr, ALIGNMENT, aligned_size) != 0) {
        return NULL;
    }
    return ptr;
}

void nvme_free_aligned(void *ptr) {
    if (ptr) free(ptr);
}

double nvme_stream_write_direct(const char *path, const void *buf, size_t size, int num_threads) {
    if (!path || size == 0) return -1.0;
    if (num_threads <= 0) num_threads = 8;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    int flags = O_RDWR | O_CREAT | O_TRUNC | O_DIRECT | O_NOATIME;
    int fd = open(path, flags, 0644);
    if (fd < 0) return -2.0;

    double t0 = get_wall_time();

    pthread_t th[MAX_THREADS];
    worker_arg_t args[MAX_THREADS];
    size_t block_align = 4096;
    size_t per_th = ((size / num_threads + block_align - 1) / block_align) * block_align;
    bool ring = (buf == NULL);

    uint8_t *ring_buffers[MAX_THREADS];
    if (ring) {
        for (int i = 0; i < num_threads; i++) {
            ring_buffers[i] = (uint8_t *)nvme_alloc_aligned(DEFAULT_CHUNK_SIZE);
            if (ring_buffers[i]) memset(ring_buffers[i], 0x5A, 4096);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        size_t off = i * per_th;
        if (off >= size) {
            args[i].size = 0;
            continue;
        }
        size_t th_size = (off + per_th > size) ? (size - off) : per_th;
        args[i].fd = fd;
        args[i].buf = ring ? ring_buffers[i] : ((uint8_t *)buf + off);
        args[i].offset = off;
        args[i].size = th_size;
        args[i].chunk_size = DEFAULT_CHUNK_SIZE;
        args[i].ring_mode = ring;
        pthread_create(&th[i], NULL, stream_write_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        if (i * per_th < size) {
            pthread_join(th[i], NULL);
        }
        if (ring && ring_buffers[i]) {
            nvme_free_aligned(ring_buffers[i]);
        }
    }

    double t1 = get_wall_time();
    close(fd);
    return (t1 - t0);
}

double nvme_stream_read_direct(const char *path, void *buf, size_t size, int num_threads) {
    if (!path || size == 0) return -1.0;
    if (num_threads <= 0) num_threads = 16;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    int flags = O_RDONLY | O_DIRECT | O_NOATIME;
    int fd = open(path, flags);
    if (fd < 0) return -2.0;

    double t0 = get_wall_time();

    pthread_t th[MAX_THREADS];
    worker_arg_t args[MAX_THREADS];
    size_t block_align = 4096;
    size_t per_th = ((size / num_threads + block_align - 1) / block_align) * block_align;
    bool ring = (buf == NULL);

    uint8_t *ring_buffers[MAX_THREADS];
    if (ring) {
        for (int i = 0; i < num_threads; i++) {
            ring_buffers[i] = (uint8_t *)nvme_alloc_aligned(DEFAULT_CHUNK_SIZE);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        size_t off = i * per_th;
        if (off >= size) {
            args[i].size = 0;
            continue;
        }
        size_t th_size = (off + per_th > size) ? (size - off) : per_th;
        args[i].fd = fd;
        args[i].buf = ring ? ring_buffers[i] : ((uint8_t *)buf + off);
        args[i].offset = off;
        args[i].size = th_size;
        args[i].chunk_size = DEFAULT_CHUNK_SIZE;
        args[i].ring_mode = ring;
        pthread_create(&th[i], NULL, stream_read_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        if (i * per_th < size) {
            pthread_join(th[i], NULL);
        }
        if (ring && ring_buffers[i]) {
            nvme_free_aligned(ring_buffers[i]);
        }
    }

    double t1 = get_wall_time();
    close(fd);
    return (t1 - t0);
}
