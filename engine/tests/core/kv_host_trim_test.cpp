// K21 host-KV trim test (4C-i / 4C-ii / 4B-lite).
//
// Two parts:
//   * a pure-CPU part (always runs, no GPU): session_needs_host's sizing decision — the 4C-i pre-pin predicate
//     that tells conv_floor_evict how many bytes a lazy host pin is about to charge.  A fabricated streamed state
//     exercises every branch (needs a pin / already held / not streamed / empty extent) and the exact byte figure.
//   * a GPU part (compiled now, executed in Task 4 beside the parity batteries): session_ensure_host -> write ->
//     session_release_host -> host.present()==false, g_kv_host_bytes back to baseline, identity page_table,
//     re-ensure backfill bit-exact, double-release idempotent, and the STRATA_KV_HOST_REGISTER=1 release path.
//
// `--cpu-only` runs just the pure-CPU part (the offline Task 3 gate, while live serving holds the GPUs).
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// ---- pure-CPU: session_needs_host (4C-i) ------------------------------------------------------------------
int cpu_part() {
    ModelGeometry g;                       // the default flash_next geometry: valid for qsa_shapes / kv_block_bytes
    QsaState st{};
    st.kv_mode = 1;                        // streamed
    st.n_pages = 8;
    st.n_slots = 2;
    SessionState ss;
    ss.qsa_states = &st;
    ss.qsa_alloc = 1;
    ss.qsa_ord0 = 0;
    ss.max_cells = 4096;

    const strata::kernels::QsaShapes s = qsa_shapes(g);
    const uint64_t per_state = (uint64_t) st.n_pages * strata::kernels::kv_block_bytes(s, qsa_kv_format(st)) + 4 * 256;

    // a streamed layer with no host copy at a positive extent needs a pin, and the figure matches the ensure path
    uint64_t pin = 12345;
    CHECK(session_needs_host(ss, g, 2048, &pin));
    CHECK(pin == per_state);
    // null out-param is allowed
    CHECK(session_needs_host(ss, g, 2048, nullptr));
    // an empty extent never needs a pin
    pin = 999;
    CHECK(!session_needs_host(ss, g, 0, &pin));
    CHECK(pin == 0);
    // a layer that already holds its host copy needs no pin (present() is a non-null check; no deref here)
    st.host.k_pool = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(1));
    pin = 999;
    CHECK(!session_needs_host(ss, g, 2048, &pin));
    CHECK(pin == 0);
    st.host.k_pool = nullptr;
    // a fully-resident (mode 0) layer never needs a host copy
    st.kv_mode = 0;
    CHECK(!session_needs_host(ss, g, 2048, &pin));
    st.kv_mode = 1;
    // several layers: the figure sums each state that still needs a pin
    QsaState states[3]{};
    for (int i = 0; i < 3; ++i) { states[i].kv_mode = 1; states[i].n_pages = 8; states[i].n_slots = 2; }
    states[1].host.k_pool = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(1));   // already held: skipped
    SessionState ss3;
    ss3.qsa_states = states; ss3.qsa_alloc = 3; ss3.qsa_ord0 = 0; ss3.max_cells = 4096;
    pin = 0;
    CHECK(session_needs_host(ss3, g, 2048, &pin));
    CHECK(pin == 2 * per_state);
    // qsa_ord0 offsets into the global array
    QsaState big[5]{};
    big[3].kv_mode = 1; big[3].n_pages = 8; big[3].n_slots = 2;
    SessionState ss_off;
    ss_off.qsa_states = big; ss_off.qsa_alloc = 1; ss_off.qsa_ord0 = 3; ss_off.max_cells = 4096;
    pin = 0;
    CHECK(session_needs_host(ss_off, g, 2048, &pin));
    CHECK(pin == per_state);

    if (failed) { std::fprintf(stderr, "kv_host_trim_test(cpu): %d failed\n", failed); return 1; }
    std::printf("kv_host_trim_test(cpu): ok\n");
    return 0;
}

// ---- GPU: the ensure / release / re-ensure cycle (Task 4) --------------------------------------------------
int gpu_part() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("kv_host_trim_test(gpu): skipped (no CUDA device)\n");
        return 77;
    }
    (void) cudaSetDevice(0);
    // streamed + lazy host: the boot path leaves the host copy unpinned, session_ensure_host pins it on demand
    setenv("STRATA_KV_HOST_LAZY", "1", 1);
    qsa_set_kv_resident(1024);
    ModelGeometry g;
    const int64_t max_cells = 4096;
    const int64_t upto = 2048;
    const uint64_t arena_bytes = qsa_state_bytes(g, max_cells);
    void* arena = nullptr;
    if (cudaMalloc(&arena, arena_bytes) != cudaSuccess) { std::fprintf(stderr, "FAIL gpu: arena malloc\n"); return 1; }
    QsaState st{};
    if (qsa_state_init(g, max_cells, arena, st) == 0) { std::fprintf(stderr, "FAIL gpu: qsa_state_init\n"); cudaFree(arena); return 1; }
    CHECK(st.kv_mode == 1);
    SessionState ss;
    ss.qsa_states = &st; ss.qsa_alloc = 1; ss.qsa_ord0 = 0; ss.max_cells = max_cells;
    strata::kernels::kv_stream_init_identity(st.map, nullptr);
    (void) cudaDeviceSynchronize();

    const uint64_t base_host = qsa_kv_host_bytes();
    const strata::kernels::QsaShapes s = qsa_shapes(g);
    const int64_t resident_rows = st.n_slots * s.page_size * s.n_head_kv * s.head_dim;   // F16 k_pool elements
    const size_t resident_bytes = (size_t) resident_rows * sizeof(uint16_t);

    // a known byte pattern in the resident VRAM k_pool; the host backfill must reproduce it bit-exact
    if (st.k_pool != nullptr && resident_bytes > 0)
        (void) cudaMemset(st.k_pool, 0xAB, resident_bytes);
    (void) cudaDeviceSynchronize();

    auto host_resident_matches = [&]() -> bool {
        if (st.host.k_pool == nullptr || st.k_pool == nullptr || resident_bytes == 0) return true;
        std::vector<uint8_t> hb(resident_bytes), vb(resident_bytes);
        std::memcpy(hb.data(), st.host.k_pool, resident_bytes);                 // host copy is pinned, readable
        if (cudaMemcpy(vb.data(), st.k_pool, resident_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        return std::memcmp(hb.data(), vb.data(), resident_bytes) == 0;
    };

    // ensure: pins the host copy and backfills the resident blocks
    bool alloc = false; std::string err;
    CHECK(session_ensure_host(ss, g, upto, alloc, err));
    CHECK(alloc);
    CHECK(st.host.present());
    CHECK(st.host_base != nullptr);
    CHECK(st.host_bytes > 0);
    CHECK(qsa_kv_host_bytes() > base_host);
    CHECK(host_resident_matches());

    // release: host gone, bytes back to baseline, VRAM untouched, page_table identity
    bool freed = false;
    CHECK(session_release_host(ss, g, err, freed));
    CHECK(freed);
    CHECK(!st.host.present());
    CHECK(st.host_base == nullptr);
    CHECK(st.host_bytes == 0);
    CHECK(!st.host_registered);
    CHECK(qsa_kv_host_bytes() == base_host);
    if (st.page_table != nullptr && st.n_slots > 0) {
        std::vector<int32_t> pt((size_t) st.n_slots);
        if (cudaMemcpy(pt.data(), st.page_table, (size_t) st.n_slots * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess)
            for (int64_t i = 0; i < st.n_slots; ++i) CHECK(pt[(size_t) i] == (int32_t) i);
    }
    CHECK(host_resident_matches());   // VRAM resident blocks unchanged by the release

    // double release is idempotent: nothing held, nothing freed, still true
    freed = true;
    CHECK(session_release_host(ss, g, err, freed));
    CHECK(!freed);
    CHECK(qsa_kv_host_bytes() == base_host);

    // re-ensure: a fresh host copy backfilled bit-exact from the untouched VRAM
    alloc = false;
    CHECK(session_ensure_host(ss, g, upto, alloc, err));
    CHECK(alloc);
    CHECK(st.host.present());
    CHECK(host_resident_matches());
    freed = false;
    CHECK(session_release_host(ss, g, err, freed));
    CHECK(freed);
    CHECK(qsa_kv_host_bytes() == base_host);

    // STRATA_KV_HOST_REGISTER=1: the host copy is pinned with cudaHostRegister over an mmap; release must
    // cudaHostUnregister + munmap it (host_registered true), and the bytes still come back to baseline.
    setenv("STRATA_KV_HOST_REGISTER", "1", 1);
    {
        QsaState st2{};
        void* arena2 = nullptr;
        if (cudaMalloc(&arena2, arena_bytes) == cudaSuccess && qsa_state_init(g, max_cells, arena2, st2) != 0) {
            SessionState ss2;
            ss2.qsa_states = &st2; ss2.qsa_alloc = 1; ss2.qsa_ord0 = 0; ss2.max_cells = max_cells;
            strata::kernels::kv_stream_init_identity(st2.map, nullptr);
            (void) cudaDeviceSynchronize();
            bool a2 = false;
            CHECK(session_ensure_host(ss2, g, upto, a2, err));
            CHECK(a2);
            CHECK(st2.host.present());
            CHECK(st2.host_registered);                 // the register-first path
            const uint64_t held = qsa_kv_host_bytes();
            CHECK(held > base_host);
            bool f2 = false;
            CHECK(session_release_host(ss2, g, err, f2));
            CHECK(f2);
            CHECK(!st2.host.present());
            CHECK(st2.host_base == nullptr);
            CHECK(qsa_kv_host_bytes() == base_host);
            cudaFree(arena2);
        }
    }
    unsetenv("STRATA_KV_HOST_REGISTER");
    cudaFree(arena);
    (void) cudaDeviceSynchronize();
    if (failed) { std::fprintf(stderr, "kv_host_trim_test(gpu): %d failed\n", failed); return 1; }
    std::printf("kv_host_trim_test(gpu): ok\n");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu-only") == 0;
    const int c = cpu_part();
    if (c != 0 || cpu_only) return c;
    const int g = gpu_part();
    if (g == 77) return 0;   // no GPU: the CPU part passed, the GPU part is a stated skip (Task 4 runs it)
    return g;
}