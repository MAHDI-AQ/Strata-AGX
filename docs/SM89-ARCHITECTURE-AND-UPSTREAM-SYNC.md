# Strata-AGX: Ada Lovelace (`sm_89`) Architecture & Upstream Sync Protocol

Guide for contributors and automated agents working on `MAHDI-AQ/Strata-AGX` (`v0.2.16` / engine `0.1.62`, zero-GGML) targeting NVIDIA Ada Lovelace (`sm_89`) multi-GPU systems through **Wave K26 Phase 1**.

---

## 1. Fork Identity & Upstream Relationship

- **Fork Repository:** `MAHDI-AQ/Strata-AGX`
- **Upstream Attribution:** Clean selective port onto [`Niko1221/Strata`](https://github.com/Niko1221/Strata) (`v0.2.16`, engine `0.1.62`, zero-GGML modular layout: `engine/`, `terminal/`, `app/`).
- **`main == green` Invariant:** `main`, annotated release tags (`v0.2.16-agx-k26p1`), and the `green` tag stay aligned at the latest verified commit. Feature and lever branches (`feat/*`, `perf/*`, `lever/*`) branch from `main` and promote back only after passing offline parity/sanitizer/spill checks, CI checks, and interleaved live qualification.
- **Standalone Hardware Focus:** All `sm_89` optimizations live in standard C++/CUDA and Python serving code behind opt-in switches (`terminal/data/settings-register.json`, `docs/SETTINGS.md`), with zero external orchestrator dependencies. Default upstream execution remains byte-identical when opt-in flags are unset or `0`.

---

## 2. SM89-Priority Guardrails, Memory Law & Rejected Switches

- **Target Hardware:** NVIDIA Ada Lovelace (`sm_89`) GPUs — `1×`, `2×`, or `4×` RTX 4090 / RTX 40-series (`24 GiB` GDDR6X) or L40S (`48 GiB` GDDR6) with PCIe Gen4 NVMe `O_DIRECT` session storage.
- **Device Ordering & Build Toolchain:**
  - Always export `CUDA_DEVICE_ORDER=PCI_BUS_ID` on heterogeneous-slot multi-GPU systems (preventing `FASTEST_FIRST` PCI-order swaps against NVML/cooling indices).
  - Configure builds from repository root with `cmake -S engine -B build -DCMAKE_CUDA_ARCHITECTURES="86;89"`. On Linux toolchains with modern glibc, pin CUDA 12.4 + `g++-13` (CUDA 13.1 `crt/math_functions.h` `rsqrtf` `noexcept` conflicts with glibc headers).
- **Session Persistence Architecture:**
  - Radix trees, paged KV, HiCache, and SGLang-style prefix sharing are excluded. The architecture is **Session Lease + in-place delta appending + `O_DIRECT` NVMe streaming** (`STRATA_CONVERSATION_CACHE_DIR`).
  - Q2 / K4V2 KV quantization and full-pipeline Ada FP8 decode are excluded in favor of Q4_0 / Q8_0 KV (`--kv q4_0` or `--kv q8_0`) plus opt-in surgical FP8 prefill, stage handoff, and FP8 LM head (`STRATA_LM_HEAD_FP8_FAST=1`).
- **VRAM & Host Memory Floors (`4× 24 GiB` `8×512k`):**
  - Keep `--resident-experts-only` (`24,576 / 24,576` experts in VRAM, `0.0 MB` runtime expert spills) and `--prefill 4096`.
  - Keep `>= 600 MiB` operational (`>= 550 MiB` hard floor) free VRAM on all 4 GPUs through isolated `400k` boundary crossings.
  - Keep `--ple-io direct --ple-row-cache 16777216` (`STRATA_SSD_KEEPALIVE=100`, `STRATA_THP_DISABLE=1`, `STRATA_DROP_GGUF_PAGECACHE=1`, `STRATA_CONV_NO_MLOCK=1`, `STRATA_CONV_MADV_DONTNEED=1`) and cgroup `memory.high=max` so `VmSwap == 0 kB` and `VmLck` drops from `~27.5 GiB` to `< 1 GiB`.
- **Permanently Rejected / Disabled Switches (`KEEP OFF`):**
  - `STRATA_SLOT_READ=1` (burst regression + stale `0%` needle bug), `STRATA_KV_COPY_RESIDENT=1` (post-`400k` repetition collapse with lazy host KV), `STRATA_PF_W8_CACHE_MIB=256` (warm-delta penalty), `STRATA_TOPK_SPLIT=1` (`c4` `-11%..-27%` regression), `STRATA_PREFILL_HELP=1` (`-38%` MPD), `STRATA_ATTN_NOSPLIT_DEEP=1` (`-2.74%` vs high-chunk gating), `STRATA_MTP_STAGES=2,3` (`-3.46%` MPD), `--mtp-window 8192` (`burst196 -18.19%`), `STRATA_GU_SWIGLU=1`, `STRATA_PIPED_SUFFIX=1`, `STRATA_EMBED_VRAM=1` (`1/8` cached needles -> lock `0`), `STRATA_MTP_DIRECT_RIN` (`0.25%` keep), `STRATA_READ_WIN_MAX_T=1` on multi-stage (`0/8`), `STRATA_TOPK_DECODE_SAMPLE=1` at `262k` (RTX 4090 `72 MiB` L2 already holds `~2 MiB` `262k` scores).

---

## 3. `1×` / `2×` / `4×` Ada Lovelace (`sm_89`) Flag Profiles

| Topology | Profile | Slots | Max Context | Resident KV / Slot | Pipeline Split Flags | Notes |
|---|---|---|---|---|---|---|
| **4× 24 GiB** (`96 GiB`) | **`8×512k @ 288k` (Locked Default)** | `--batch 8` | `--max-context 524288` | `--kv-resident 294912` | `--layer-split 14,27,39 --split-device 3,2,1` | `>= 600 MiB` free floor (`950 MiB` at `400k`), `24,576 MiB` / `16` slots conv cache |
| **4× 24 GiB** (`96 GiB`) | `4×1m @ 128k` (Optional High-Context) | `--batch 4` | `--max-context 1048576` | `--kv-resident 131072` | `--layer-split 14,27,39 --split-device 3,2,1` | `--rope-scale 4.0`, `--yarn-orig-ctx 262144` |
| **2× 24 GiB** (`48 GiB`) | `4×256k @ 64k` | `--batch 4` | `--max-context 262144` | `--kv-resident 65536` | `--layer-split 24 --split-device 1,0` | Balanced 2-card split |
| **2× 24 GiB** (`48 GiB`) | `8×128k @ 32k` | `--batch 8` | `--max-context 131072` | `--kv-resident 32768` | `--layer-split 24 --split-device 1,0` | 8-slot 2-card split |
| **1× 24 GiB** (`24 GiB`) | `4×128k @ 32k` | `--batch 4` | `--max-context 131072` | `--kv-resident 32768` | *(single GPU, no split)* | Single-card 4-slot |
| **1× 24 GiB** (`24 GiB`) | `8×64k @ 16k` | `--batch 8` | `--max-context 65536` | `--kv-resident 16384` | *(single GPU, no split)* | Single-card 8-slot |

Core `4× sm_89` production flags (`8×512k` default):
`--resident-experts-only --ple-io direct --ple-row-cache 16777216 --prefill 4096 --kv q4_0 --rope-scaling yarn --rope-scale 2.0 --yarn-orig-ctx 262144 --batch 8 --batch-groups auto --batch-mtp --batch-mtp-mode shared --group-spread-mtp --concurrent-prefill --mtp-q4 all --spec 4 --mtp-max-t 4 --suffix-draft 3 --spec-min-p 0.50 --mtp-window 32768 --conversation-cache-mib 24576 --conversation-cache-slots 16 --share-batch-rope --no-solo-buffers --parallel-stage-checkpoints --aux-cpus 1-7,17-23`

---

## 4. Complete Wave K1–K26 Phase 1 Optimization Map

| Wave | Primary Files | Implemented Switches / Fixes | Status |
|---|---|---|---|
| K1–K6 | `engine/src/prefill/gemm.cu`, `engine/src/kernels/cuda/qsa_select.cu` | `STRATA_PF_FP8_PER_TOKEN=1`, `STRATA_PF_FP8_PAIR_TICKET=1`, `STRATA_QSA_HALVE_SM89=1`, `STRATA_BATCH_DOORBELL=1`, `STRATA_QFUSE=1`, `STRATA_SELECT_TWO_PASS=1` | Active |
| K7–K10 | `engine/src/core/verify.cpp`, `engine/src/core/mtp.cpp`, `engine/src/kernels/cuda/{sampler,native_mmvq,verify_kernels}.cu` | `STRATA_QSA_SLOT_RESOLVE=1`, `STRATA_MTP_FUSED_CHAIN=1` + coupled, zero-spill `sampler`, `IlQ6K`, `gdn_out_norm`, `evict_batch_graph` handle fix, `STRATA_BATCH_GRAPH_CAP=64` | Active |
| K11–K13 | `terminal/serve/{server,nvme_session_streamer}.py`, `engine/src/core/{weights,native_head}.cpp`, `engine/src/program/generate.cpp` | `STRATA_BINARY_IPC=1` (`BGENB`), `STRATA_CONV_NO_MLOCK=1`, `STRATA_CONV_MADV_DONTNEED=1`, `STRATA_SKIP_SHORT_PARK_MAX=2048`, `STRATA_BYIELD_SUBCHUNK=1024`, `STRATA_L2_PERSIST=1`, `STRATA_PREFILL_SMALL_T_RESIDENT=1`, `STRATA_EMBED_VRAM=0` | Active |
| K14–K16 | `engine/src/core/verify.cpp`, `engine/src/kernels/cuda/{kv_stream,qsa_select,sampler}.cu`, `engine/src/prefill/gemm.cu`, `terminal/serve/server.py` | `STRATA_SAMPLER_BATCH_ROWS=1`, `STRATA_SAMPLER_ASYNC=1`, `STRATA_KV_STAGE_SEL_GPU=1`, `STRATA_GDN_NORM_QFUSE=1`, `STRATA_GDN_CONV_GROUPS=1`, `STRATA_PARK_STAGE_THREADS=1`, `STRATA_PARK_STAGE_POOL=1`, `STRATA_PARK_SEGPOOL_MIB=0`, `STRATA_TOPK_ACTIVE_SM89=1`, `STRATA_KV_STREAM_WIDE=1`, `STRATA_PREFILL_LOW_PRIO=1`, `STRATA_MLOCK_CODE=1`, `STRATA_DROP_STARTUP_MMAP=1`, `STRATA_PF_FP8_MIN_T=32` | Active |
| K17–K18 | `engine/include/strata/core/conversation_*.hpp`, `engine/src/prefill/{gemm.cu,prefill.cpp}`, `engine/src/core/verify.cpp`, `terminal/serve/server.py` | `STRATA_CONV_JEMALLOC_PURGE=1`, `STRATA_CONV_NOHUGEPAGE=1`, `STRATA_MSG_PREFIX_CACHE=2` (`orjson`), `STRATA_PF_FP8_PAD_SMALL_T=1`, `STRATA_PF_SMALL_T_RUN=1`, `STRATA_PF_SMALL_T_RESIDENT_STAGED=1`, `STRATA_PF_RESIDENT_SMALL_T=1`, `STRATA_PF_FP8_SMALL_T_REUSE=1`, `STRATA_SLOT_COPY_STREAM_SYNC=1`, `STRATA_ASYNC_STAGE_WAIT=1`, `STRATA_READ_WIN_NO_HEAD=1`, `STRATA_CONV_CACHE_MEM_FLOOR=1`, `Verifier::record_window` fix | Active |
| K19 | `engine/src/core/{layer,verify,conversation_snapshot,conversation_state}.cpp`, `engine/src/kernels/cuda/{kv_stream,qsa_select}.cu` | `STRATA_QSA_IDX_FP16=1`, `STRATA_KV_HOST_LAZY=1` (`0 GiB` boot RAM, VRAM D2D `1.8–2.3 ms`), `STRATA_KV_RESOLVE_BATCH=1`, `STRATA_IPC_DELTA=1`, `STRATA_ROPE_TABLE=1`, `STRATA_SESSION_HASH=1`, `STRATA_GDN_QKVZ=1`, `zlib.crc32` `BGEND` | Active |
| K20a–K20c | `engine/src/{core/layer.cpp,program/generate.cpp}`, `terminal/serve/server.py` | Selective upstream `v0.2.16` (`engine 0.1.62` zero-GGML) port, `STRATA_KV_STAGE_LAZY=1`, `STRATA_PROMPT_ATTN_SPLIT_KV=1`, `STRATA_PICK_SLOT_GROUP_SPREAD=1`, `STRATA_THP_DISABLE=1`, `STRATA_DROP_GGUF_PAGECACHE=1`, `STRATA_ADMIT_FIFO=0`, `STRATA_BATCH_SLOT_RUNS=1`, `STRATA_SLOT_CKPT_KV_DIRECT=1`, `STRATA_PIPED_LOOKUP=1` | Active |
| K21–K24 | `engine/src/core/{conversation_snapshot,conversation_state}.cpp`, `engine/src/core/layer.cpp`, `terminal/serve/{server,nvme_session_streamer}.py` | Eviction-park cold re-park fallback (`conversation_kv_prefix_holds`, `d5d13aadd`), `STRATA_VISION_LAZY=1` / `STRATA_VISION_IDLE_S=300` + tool-image `last_image_at` fix (`e41fdf94d`), `STRATA_SSD_KEEPALIVE=100` (`--ple-io direct`), host-KV floor evict (`80880cbde`), pooled-indexer byte-width retention fix (`68451d330`), 2 MiB NVMe header overflow guard (`89fb8339d`) | Active |
| K25b & CI | `engine/include/strata/kernels/native_head_gemv.cuh`, `engine/src/core/native_fp8.cu`, `tools/check`, `.github/workflows/check.yml`, `terminal/tools/test_setup_sycl.py` | `STRATA_ATTN_MERGE_V2=1`, `STRATA_LM_HEAD_FP8_FAST=1` (XOR-swizzled SMEM + warp-4 parallel argmax, `!gumbel_enabled`, `742a2b882`), LPT CI sharding + path-filtered `Checks` (`28m34s -> 7m49s` / `3m34s`) + explicit `setup_intel.py` import | Active |
| K26 Ph 1 | `engine/src/core/mtp.cpp`, `engine/src/kernels/cuda/{qsa_select,qsa_decode_attn}.cu` | `STRATA_MTP_Q4_SHARED=1` (`3706c4d34`, `dense4_`/`q4_off_` slot sharing + `owns_weights_` dtor guard), `STRATA_QSA_ROWS_RUNS=1` (`3be67f536`, same-slot run split `<= 4`), `STRATA_ATTN_NOSPLIT_MIN_CHUNKS=64` (`4f073259f`, `SPLIT_HEAD=false` for `!rot_q && n_chunks >= 64`) | Active |

---

## 5. Verification Gate & Benchmarking Discipline

1. **CUDA Spill Gate:**
   ```sh
   python3 tools/check_cuda_spills.py build/libstrata_kernels.a build/libstrata_core.a build/libstrata_engine.a
   python3 -m unittest tools.test_check_cuda_spills
   ```
   All `sm_89` kernels must report `0 not allowed`.
2. **Offline Unit, Registry & Docs Suite:**
   ```sh
   python3 tools/check --only docs
   python3 -m unittest tools.test_settings_register
   python3 -m unittest discover -s terminal/tools -p 'test_uma_drop_guard.py'
   python3 -m pytest terminal/serve/test_server.py
   ```
   After `python3 tools/settings_register.py --docs`, verify `## Serve-side image reader switches (Terminal)` (`STRATA_VISION_LAZY`, `STRATA_VISION_IDLE_S`) remains at the bottom of [SETTINGS.md](SETTINGS.md).
3. **Idle-GPU Parity & Sanitizer:**
   Run affected `build/*_parity` and `build/*_test` targets (`batch_groups_test`, `batch_read_test`, `batch_rows_parity`, `kv_window_parity`, `gemm_fp8_parity`, `sampler_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity`) under `compute-sanitizer --tool memcheck` (`0 errors`) with `0` bit/token mismatches. Note `vram_oc_probe` / multi-iteration parity (`~60–150` reps) catches intermittent DRAM flips that a 1-shot run misses.
4. **Live Interleaved `A1 / B / A2` & Isolation Discipline:**
   - Gate every live optimization with interleaved `A1 / B / A2` using fresh session IDs, normalizing stochastic (`temperature=0.7`) burst comparisons by `completion_tokens`, `agg_decode_tok_s`, `ttft_p50`, and CUPTI completed verify steps rather than raw `turns/min` alone.
   - Always run isolated extreme-context boundary crossings (`400k+`) separately from cumulative `gate + burst65 + burst196` runs on the same process (or after a clean restart) so cumulative batch CUDA graph pools plus lazy staging do not squeeze Stage 1 (`CUDA3`) below the safe driver margin.
   - Qualify conversation cache eviction pressure with rotation counts strictly larger than `--conversation-cache-slots` (`20 > 16`, asserting `evictions > 0` and `0` missing prefixes).

---

## 6. Upstream Sync Protocol (`Niko1221/Strata` `v0.2.16`+)

1. Audit divergence before integrating upstream changes:
   ```sh
   git fetch upstream && git log HEAD..upstream/main --oneline
   python3 terminal/tools/upstream_sync_audit.py --repo . --upstream-ref upstream/main --fetch
   ```
2. Port onto a staging branch off `main` (`green`), preserving `engine/include/strata/core/nv_defaults.hpp`, `--vanilla-defaults`, and the `STRATA_READ_WIN_MAX_T=0` invariant (keeping `STRATA_READ_WIN_MAX_T=1` off on multi-stage `--layer-split` prevents `T > S` misalignment at `turn_at`).
3. Keep `MAHDI-AQ/Strata-AGX` free of host-specific IPs, lab paths, or external orchestrator identifiers, then verify `tools/check --only docs`, the 3-archive CUDA spill gate, idle-GPU parity binaries, and GitHub Actions CI before advancing `main`, `green`, and release tags.

---

## 7. Measured Wave Progression (K20b → K26 Phase 1) & K26 Phase 2 Frontier

> **Measured on:** 4× RTX 4090 (`sm_89`, 96 GiB VRAM total, PCIe MRRS 4096 B, uniform +1500 MT/s memory OC / natural core boost, `<= 75°C`), Ryzen 9 5950X, 91.9 GiB RAM, PCIe Gen4 NVMe, Qwen3.8-Flash-Next `IQ3_S`, `Strata-AGX` `v0.2.16` (engine `0.1.62`, zero-GGML), `8×512k` (`--batch 8 --prefill 4096 --kv q4_0 --kv-resident 294912 --layer-split 14,27,39 --ple-io direct --conversation-cache-mib 24576 --conversation-cache-slots 16`).

### Wave K20b → K26 Phase 1 Highlights

- **Wave K20b (`03b192e2d`, `v0.2.16` port + `STRATA_PICK_SLOT_GROUP_SPREAD=1`):** `MPD_a65536_4D_4P` jumped `+30.9%` (`292.36 -> 382.67 tok/s`), `196k` burst `232.47 turns/min` (`191.30 agg_dec`).
- **Wave K20c (`90e841506`):** `STRATA_THP_DISABLE=1` + `--kv-resident 294912` (`288k`) cut `VmSwap` `5.7–7.3 GiB -> 0 kB`, `SD_r262k_c8 = 722.1 tok/s`, `400k` exact cold/warm (`min_vram_free = 606 MiB`).
- **Wave K21 (`496f9d597`):** `--ple-io direct --ple-row-cache 16777216` cut `VmLck 27.55 -> 0.72 GiB` and freed `+26.8 GiB` host RAM (`MemAvail 30.7 -> 57.5 GiB`) while lifting `burst65` `+4.3%`; eviction-park guard + `STRATA_VISION_LAZY=1` + `24 GiB / 16` slots sealed.
- **Wave K22–K24 (`68451d330`..`89fb8339d`):** Fixed pooled-indexer `fp16` retention bug (`88/88` cold re-park elimination -> warm resume TTFT `5,483 -> 2,174 ms`, `9.12×` vs cold), natural boost (`-rgc`, `P8 72.7 W` idle), and `nvme_session_streamer.py` 2 MiB header overflow fix at `~393k` tokens; 60-minute endurance soak completed `2,003` turns / `42.7M` prompt tokens with `0` errors and `3× 40/40` clean rotations.
- **Wave K25b (`742a2b882`..`a7c8138d2`):** `STRATA_ATTN_MERGE_V2=1` (`SD_r262k_c8 +5.47%`) + `STRATA_LM_HEAD_FP8_FAST=1` (`1.31–1.68×` offline LM head, `+3.57%` live ABA -> **`+8.02%` cumulative** `675.61 -> 729.79 tok/s`, GPU 1 `-738 ms` kernel sum, `MPD = 380.67 tok/s`), plus `8.0×` faster CI (`28m34s -> 3m34s` path-filtered).

### Wave K26 Phase 1 Sealed Result (`4f073259f`)

- **Full Qualification (`4f073259f`, all checks PASS):**
  - Cold Smoke (`4×5k`): `100.0% / 0.0% / 100.0%`, `781.3 ms` mean TTFT (`66.1 tok/s` mean decode).
  - Fast Gate (`--slots 8`): `SD_r512_c8 = 723.98`, `r32k_c8 = 769.46`, `r131k_c8 = 737.50`, `r262k_c1 = 155.07`, `r262k_c4 = 539.27`, `r262k_c8 = 716.39 tok/s`, `MPD_a65536 = 372.06 tok/s`, `NEEDLE_GATE 8/8`, `0` leaks, `min_vram_free = 912 MiB`.
  - Agentic Bursts: `burst65` `259.95 turns/min` (`228.09 agg_decode_tok_s`, `ttft_p50 = 1341.36 ms`, `8/8` needles); `burst196` `254.33 turns/min` (`180.95 agg_decode_tok_s`, `ttft_p50 = 1460.16 ms`, `8/8` needles).
  - Isolated `400k` Cold + Warm: `exact_match = True / True` (`15,839.7 tok/s` cold prefill, `[1254, 1930, 1896, 950] MiB` free -> `950 MiB` min free), post-`400k` smoke `100.0% / 0.0% / 100.0%` (`748.5 ms` TTFT, `90.3 tok/s`).
  - Fleet Rotations (`2× 20×65k`): `20/20` both rounds across both trials, `0` leaks, `0` missing prefixes (`evictions = 16, 40`).
  - Offline & CI: `421` server pytest (`576` subtests), `cuda_spill_guard` PASS, `tools/check --only docs` `6/6`, GitHub Actions `3/3` green.
- **Hardware `nsys` Profile Delta (`12.08s` window, `8 × 262k`, K25b → K26 Phase 1):**
  - Completed verify steps (`fused_lm_head` calls): `1,708 -> 1,765` (`+3.34%`).
  - Bottleneck Stage 3 (`GPU 1`): union busy `95.65% -> 94.04%`, kernel sum `15.630s -> 15.017s` (`-7.0%` work per step), allowing GPUs 0/2/3 utilization to rise (`76.28% -> 77.13%`, `67.45% -> 68.13%`, `70.83% -> 71.35%`).
  - `native_small_mmvq_kernel<Q80Block>`: `74,973` calls (`991.70 ms`) -> `0` calls (`Q80Block` total `-99.27%` to `8.745 ms`); combined `Q80 + Q40` MTP MMVQ `-15.5%` per step.
  - `block_scores_multi_kernel` (4 GPUs): `20,504` calls (`2,108.57 ms`) -> `0` calls, replaced by `block_scores_rows_kernel<2..4, 1>` (`1,397.02 ms`, `-33.7%` total).
  - `attn_chunk_kernel` (GPU 1): `100%` switched to `SPLIT_HEAD=0` (`136.2 -> 129.5 us/call`, `-5.0%`/call).

### Ranked Wave K26 Phase 2 Frontier Targets (on GPU 1, Still `94.04%` Busy)

1. **Batched 2-slot `T=2` MTP projections (`native_small_mmvq_kernel<Q40Block, 4, false>` — `2,507.4 ms`, `15.3%+` GPU 1):** pair the 2 slots per batch group (`dhead_` + projections with `T=2`) to halve draft weight streaming bandwidth.
2. **GDN linear recurrence & split norms (`gr_down` + `gr_up` + `gr_norm_split` — `1,949 ms`, `13.0%` GPU 1).**
3. **QSA attention & value merge (`attn_chunk_kernel` `1,371.4 ms` + `attn_merge_v2` `428.7 ms` + `attn_chunk_rows` `380.0 ms`, `14.5%` GPU 1).**
4. **In-place `Q6_K` body packing (`STRATA_Q6_BODY_PACKED`) + MoE `native_down`/`gu` (`1,339.8 ms` GPU 1);** note `fused_lm_head_gemv_fast_argmax_kernel` (`1,196.6 ms`, `8.0%`) is already near the DRAM ceiling.
5. **Multi-pass `block_topk_wide_kernel` (`567.4 ms` GPU 1 / `2,230.3 ms` 4-GPU)** at `>= 262k` context (noting L2 holds `~2 MiB` scores at `262k` on `72 MiB` L2 GPUs).
