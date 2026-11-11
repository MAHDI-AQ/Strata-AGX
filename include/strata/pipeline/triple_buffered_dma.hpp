#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define STRATA_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + \
                                     std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while (0)

namespace strata {
namespace pipeline {

/**
 * TripleBufferedDMAPipeline
 * 
 * 3-slot lockless ring-buffer DMA pipeline orchestrator.
 * Mandated by Adversarial Inquisitor to eliminate Write-After-Read (WAR) hazards.
 * Chains transfers across asynchronous hardware Copy Engines on PCIe without CPU blocking.
 */
class TripleBufferedDMAPipeline {
public:
    static constexpr int SLOTS = 3;

    int dev_src;
    int dev_dst;
    size_t act_bytes;

    // 3-slot pinned host staging buffers (Accessible by all devices)
    void* h_staging[SLOTS];

    // Device memory allocations
    void* d_src_act[SLOTS];
    void* d_dst_recv[SLOTS];

    // Independent hardware streams
    cudaStream_t stream_src_compute;
    cudaStream_t stream_src_dma_tx;
    cudaStream_t stream_dst_dma_rx;
    cudaStream_t stream_dst_compute;

    // Hardware events for cross-engine, cross-device zero-CPU chaining
    cudaEvent_t ev_src_compute_done[SLOTS];
    cudaEvent_t ev_src_d2h_done[SLOTS];
    cudaEvent_t ev_dst_h2d_done[SLOTS];
    cudaEvent_t ev_dst_compute_done[SLOTS];

    TripleBufferedDMAPipeline(int src_gpu, int dst_gpu, size_t activation_bytes)
        : dev_src(src_gpu), dev_dst(dst_gpu), act_bytes(activation_bytes) {
        
        // 1. Allocate 3 pinned host staging buffers with portable mapping
        for (int i = 0; i < SLOTS; ++i) {
            STRATA_CUDA_CHECK(cudaHostAlloc(&h_staging[i], act_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
        }

        // 2. Setup Source GPU (Producer)
        STRATA_CUDA_CHECK(cudaSetDevice(dev_src));
        STRATA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_src_compute, cudaStreamNonBlocking));
        STRATA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_src_dma_tx, cudaStreamNonBlocking));
        for (int i = 0; i < SLOTS; ++i) {
            STRATA_CUDA_CHECK(cudaMalloc(&d_src_act[i], act_bytes));
            STRATA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_src_compute_done[i], cudaEventDisableTiming));
            STRATA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_src_d2h_done[i], cudaEventDisableTiming));
        }

        // 3. Setup Destination GPU (Consumer)
        STRATA_CUDA_CHECK(cudaSetDevice(dev_dst));
        STRATA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_dst_dma_rx, cudaStreamNonBlocking));
        STRATA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_dst_compute, cudaStreamNonBlocking));
        for (int i = 0; i < SLOTS; ++i) {
            STRATA_CUDA_CHECK(cudaMalloc(&d_dst_recv[i], act_bytes));
            STRATA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_dst_h2d_done[i], cudaEventDisableTiming));
            STRATA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_dst_compute_done[i], cudaEventDisableTiming));
        }
    }

    ~TripleBufferedDMAPipeline() {
        cudaSetDevice(dev_src);
        cudaDeviceSynchronize();
        cudaSetDevice(dev_dst);
        cudaDeviceSynchronize();

        for (int i = 0; i < SLOTS; ++i) {
            if (h_staging[i]) cudaFreeHost(h_staging[i]);
            if (d_src_act[i]) {
                cudaSetDevice(dev_src);
                cudaFree(d_src_act[i]);
            }
            if (d_dst_recv[i]) {
                cudaSetDevice(dev_dst);
                cudaFree(d_dst_recv[i]);
            }
            cudaSetDevice(dev_src);
            cudaEventDestroy(ev_src_compute_done[i]);
            cudaEventDestroy(ev_src_d2h_done[i]);

            cudaSetDevice(dev_dst);
            cudaEventDestroy(ev_dst_h2d_done[i]);
            cudaEventDestroy(ev_dst_compute_done[i]);
        }

        cudaSetDevice(dev_src);
        cudaStreamDestroy(stream_src_compute);
        cudaStreamDestroy(stream_src_dma_tx);

        cudaSetDevice(dev_dst);
        cudaStreamDestroy(stream_dst_dma_rx);
        cudaStreamDestroy(stream_dst_compute);
    }

    /**
     * Enqueue a microbatch pipeline step without any CPU synchronization.
     * Hardware Copy Engines and SMs are coordinated via hardware event triggers.
     */
    void enqueue_stage_transfer(int microbatch_idx) {
        const int s = microbatch_idx % SLOTS; // Ring-buffer slot: 0, 1, or 2

        // -----------------------------------------------------------------
        // Producer GPU (Stage i)
        // -----------------------------------------------------------------
        STRATA_CUDA_CHECK(cudaSetDevice(dev_src));

        // Backpressure check: Ensure consumer has completed reading slot s
        // from previous iteration (microbatch >= 3) before overwriting host memory.
        if (microbatch_idx >= SLOTS) {
            STRATA_CUDA_CHECK(cudaStreamWaitEvent(stream_src_dma_tx, ev_dst_h2d_done[s], 0));
        }

        // Hardware Copy Engine waits for SM compute kernel to finish writing d_src_act[s]
        STRATA_CUDA_CHECK(cudaStreamWaitEvent(stream_src_dma_tx, ev_src_compute_done[s], 0));

        // Asynchronous D2H DMA copy via Copy Engine to pinned host staging RAM
        STRATA_CUDA_CHECK(cudaMemcpyAsync(
            h_staging[s],
            d_src_act[s],
            act_bytes,
            cudaMemcpyDeviceToHost,
            stream_src_dma_tx
        ));

        // Signal D2H completion
        STRATA_CUDA_CHECK(cudaEventRecord(ev_src_d2h_done[s], stream_src_dma_tx));

        // -----------------------------------------------------------------
        // Consumer GPU (Stage i + 1)
        // -----------------------------------------------------------------
        STRATA_CUDA_CHECK(cudaSetDevice(dev_dst));

        // Hardware Copy Engine on consumer GPU waits for producer D2H DMA to land in host RAM
        STRATA_CUDA_CHECK(cudaStreamWaitEvent(stream_dst_dma_rx, ev_src_d2h_done[s], 0));

        // Asynchronous H2D DMA copy via Copy Engine into consumer VRAM d_dst_recv[s]
        STRATA_CUDA_CHECK(cudaMemcpyAsync(
            d_dst_recv[s],
            h_staging[s],
            act_bytes,
            cudaMemcpyHostToDevice,
            stream_dst_dma_rx
        ));

        // Signal H2D completion (allows producer to reuse host slot s on subsequent turn)
        STRATA_CUDA_CHECK(cudaEventRecord(ev_dst_h2d_done[s], stream_dst_dma_rx));

        // Consumer SM compute stream waits for H2D DMA to finish before launching kernel
        STRATA_CUDA_CHECK(cudaStreamWaitEvent(stream_dst_compute, ev_dst_h2d_done[s], 0));
    }
};

} // namespace pipeline
} // namespace strata
