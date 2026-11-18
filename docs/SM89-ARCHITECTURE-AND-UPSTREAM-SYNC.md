# Strata-AGX: Ada Lovelace (`sm_89`) Architecture & Upstream Sync Protocol

## 1. Fork Identity & Upstream Relationship

- **Fork Repository**: `MAHDI-AQ/Strata-AGX`
- **Upstream Attribution**: Forked from [`Niko1221/Strata`](https://github.com/Niko1221/Strata) (`v0.2.15`).
- **`main == green` Invariant**: `main` and the `green` tag stay aligned at the latest verified commit. Feature and optimization branches (`feat/*`, `perf/*`, `lever/*`) branch from `main` and promote back only after passing the CUDA spill gate, idle-GPU parity binaries, and serving isolation checks.

---

## 2. SM89-Priority Defaults & Guardrails

- **Target Hardware**: NVIDIA Ada Lovelace (`sm_89`) GPUs — RTX 4090 / RTX 40-series (24 GiB GDDR6X) and L40S (48 GiB GDDR6) in `1×`, `2×`, or `4×` GPU configurations with NVMe `O_DIRECT` session storage.
- **Defaults & Opt-Out**: `include/strata/core/nv_defaults.hpp` encodes `sm_89`-priority defaults; passing `--vanilla-defaults` restores upstream defaults.
- **Session Persistence Architecture**:
  - Radix trees, paged KV, HiCache, and SGLang-style prefix sharing are excluded. The sanctioned architecture is **Session Lease + in-place delta appending + `O_DIRECT` NVMe streaming** (`STRATA_CONVERSATION_CACHE_DIR`).
  - Q2 / K4V2 KV quantization and full Ada FP8 decode are excluded in favor of Q4_0 / Q8_0 KV (`--kv q4_0` or `--kv q8_0`) and FP8 per-token prefill.

---

## 3. `1×` / `2×` / `4×` Ada Lovelace (`sm_89`) Flag Profiles

| Topology | Profile | Slots | Max Context | Resident KV / Slot | Pipeline Split Flags | Notes |
|---|---|---|---|---|---|---|
| **4× 24 GiB** (96 GiB) | `8×512k @ 128k` | `--batch 8` | `--max-context 524288` | `--kv-resident 131072` | `--layer-split 13,26,39 --split-device 3,2,1` | High-concurrency default (`STRATA_EMBED_VRAM=0`) |
| **4× 24 GiB** (96 GiB) | `4×1m @ 128k` | `--batch 4` | `--max-context 1048576` | `--kv-resident 131072` | `--layer-split 13,26,39 --split-device 3,2,1` | High-context profile (`STRATA_EMBED_VRAM=0`) |
| **2× 24 GiB** (48 GiB) | `4×256k @ 64k` | `--batch 4` | `--max-context 262144` | `--kv-resident 65536` | `--layer-split 24 --split-device 1,0` | Balanced 2-card split |
| **2× 24 GiB** (48 GiB) | `8×128k @ 32k` | `--batch 8` | `--max-context 131072` | `--kv-resident 32768` | `--layer-split 24 --split-device 1,0` | 8-slot 2-card split |
| **1× 24 GiB** (24 GiB) | `4×128k @ 32k` | `--batch 4` | `--max-context 131072` | `--kv-resident 32768` | *(single GPU, no split)* | Single-card 4-slot |
| **1× 24 GiB** (24 GiB) | `8×64k @ 16k` | `--batch 8` | `--max-context 65536` | `--kv-resident 16384` | *(single GPU, no split)* | Single-card 8-slot |

Common `sm_89` serving flags across profiles:
`--batch-mtp-mode shared --group-spread-mtp --concurrent-prefill --prefill 4096 --kv q4_0` (use `--prefill 2048` on 1× 24 GiB cards if headroom is tight).

---

## 4. Wave K1–K17 Subsystem & Optimization Map

| Wave | Subsystem / Files | Optimization / Change | Status |
|---|---|---|---|
| K1–K6 | `src/prefill/gemm.cu`, `src/kernels/cuda/qsa_select.cu` | FP8 per-token prefill, pair-ticket activation (`STRATA_PF_FP8_PAIR_TICKET=1`), `STRATA_QSA_HALVE_SM89=1` | Active |
| K7–K9 | `src/core/verify.cpp`, `src/core/mtp.cpp` | `STRATA_QSA_SLOT_RESOLVE=1`, `STRATA_MTP_FUSED_CHAIN=1` + coupled, `--group-spread-mtp`, `--batch-mtp-mode shared` | Active |
| K10 | `src/kernels/cuda/sampler.cu`, `native_mmvq.cu`, `verify_kernels.cu` | Zero-spill `sampler`, `IlQ6K`, `gdn_out_norm` 128t + `float4` | Active |
| K11 | `serve/server.py`, `serve/nvme_session_streamer.py`, `src/program/generate.cpp` | `STRATA_BINARY_IPC=1` (`BGENB`), `STRATA_CONV_NO_MLOCK=1` + `MADV_DONTNEED`, `STRATA_SKIP_SHORT_PARK_MAX=2048`, `STRATA_BYIELD_SUBCHUNK=1024` | Active |
| K12 | `src/core/weights.cpp`, `src/prefill/prefill.cpp` | `STRATA_L2_PERSIST=1` (49.50 MiB L2 window), `STRATA_PREFILL_SMALL_T_RESIDENT=1` | Active |
| K13 | `native_mmvq.cu`, `sampler.cu`, `weights.cpp`, `prefill.cpp`, `native_head.cpp`, `generate.cpp`, `qsa_select.cu` | `IlQ6K NC>=7` (and `NC>=5 && R>1`) routed to `native_mmvq_il_kernel` (`STACK:0`) + `NC=5..6 R=1` `native_mmvq_il_pf_kernel` (`DB=0, MINB=4`, `STACK:0`), `sampled_tail_warp` dead `double* ex` removed + `p.temperature > 0.01f ? (1.0f / p.temperature) : 100.0f` FP32 clamp, `pack_l2_persist` atomic commit + `256B` align + `~WeightTable` free, `kv_stage_block_from_host` tail-only, `STRATA_EMBED_VRAM="0"` locked default (`NativeEmbed::ensure_device` retained defensively), parallel `ckpt_threads` `direct_slot_copy`, `#63-D` shift UB fix, dead-flag purge | Active |
| K14 | `verify.cpp`, `sampler.cu`, `generate.cpp`, `qsa_select.cu`, `kv_stream.cu`, `serve/server.py`, `tools/strata_tokenizer.py` | `STRATA_SAMPLER_BATCH_ROWS=1` batched homogeneous `Verifier::sample_rows` + `counters_d` multi-row sampler (`STACK:0`, `LOCAL:0`), multi-stage `read_windows` `win_step`/`sync` guard, lock-free `in_has_lines` `try_next_line`, `STRATA_PARK_STAGE_THREADS=1` parallel stage park, `STRATA_TOPK_ACTIVE_SM89=1` active-block QSA top-k, `STRATA_KV_STREAM_WIDE=1` 256-thread `copy_kernel`, `PromptEncoder` prefix/tokenizer locks, `TCP_NODELAY`, `STRATA_PARK_SEGPOOL_MIB=0` | Active |
| K15–K16 | `gemm.cu`, `kv_stream.cu`, `weights.cpp`, `generate.cpp`, `serve/server.py` | `STRATA_KV_STAGE_SEL_GPU=1`, `STRATA_SAMPLER_ASYNC=1`, `STRATA_GDN_NORM_QFUSE=1`, `STRATA_GDN_CONV_GROUPS=1`, `STRATA_CONC_SUBCHUNK=512`, `STRATA_PREFILL_LOW_PRIO=1`, `STRATA_PARK_STAGE_POOL=1`, `STRATA_MLOCK_CODE=1`, `STRATA_DROP_STARTUP_MMAP=1`, `STRATA_PF_FP8_MIN_T=32`, `--kv-resident 212992` | Active |
| K17 | `conversation_cache.hpp`, `conversation_buffer.hpp`, `serve/server.py`, `gemm.cu`, `prefill.cpp`, `generate.cpp`, `verify.cpp`, `mtp.cpp` | `STRATA_CONV_JEMALLOC_PURGE=1`, `STRATA_CONV_NOHUGEPAGE=1`, `STRATA_MSG_PREFIX_CACHE=1`, `STRATA_PF_FP8_PAD_SMALL_T=1`, `STRATA_PF_SMALL_T_RUN=1`, `STRATA_SLOT_COPY_STREAM_SYNC=1`, `STRATA_ASYNC_STAGE_WAIT=1` | Active |

---

## 5. Verification: 8-Object Spill Gate, Parity & Quality Isolation

1. **8-Object CUDA Spill Gate**:
   ```bash
   python3 tools/check_cuda_spills.py build/libstrata_kernels.a build/libstrata_core.a
   python3 -m unittest tools.test_check_cuda_spills
   ```
   Must report `0` disallowed spills across all 8 `.cu.o` targets (`sampler`, `native_mmvq`, `qsa_select`, `gdn_chunk`, `iq_kernels`, `gemm`, `kv_stream`, `verify_kernels`) and `OK` unit tests.
2. **8/8 Idle-GPU Parity Binaries** (run with no competing server holding VRAM):
   `batch_groups_test`, `batch_read_test`, `batch_rows_parity`, `kv_window_parity`, `gemm_fp8_parity`, `sampler_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity`.
3. **Quality Isolation & Multi-Slot Serving Smoke**:
   Run a 4-slot / 8-slot needle isolation + multi-turn reasoning check against your local OpenAI-compatible endpoint (`http://127.0.0.1:8080/v1/chat/completions`) and verify `100%` needle recall, `0%` cross-slot contamination, `0` CUDA spills, and `0` Xids (`dmesg`).

---

## 6. Upstream Sync Protocol (`Niko1221/Strata` `v0.2.15`)

1. Audit compatibility before fetching or merging:
   ```bash
   python3 tools/upstream_sync_audit.py --repo . --upstream-ref upstream/main --fetch
   ```
2. Cherry-pick or rebase upstream `v0.2.15` commits onto a staging branch off `main`.
3. Preserve `include/strata/core/nv_defaults.hpp`, `--vanilla-defaults`, and the `sm_89` Wave K1–K14 invariants.
4. Run the 8-object CUDA spill gate, 8/8 idle-GPU parity binaries, and local multi-slot quality isolation checks before fast-forwarding `main == green`.

### `STRATA_READ_WIN_MAX_T=0` Invariant on Multi-Stage `--layer-split`

Keep `STRATA_READ_WIN_MAX_T=0` (upstream default) whenever `ver.max_t() > S` (for example `--lookup-chain 3` with `--spec 4`) on multi-stage `--layer-split`. Enabling `STRATA_READ_WIN_MAX_T=1` lets `read_windows` issue `T > S` windows (e.g. `T=6` on `enable_thinking=false` turn headers), which leaves the verifier state misaligned at `turn_at`.

### Wave K14 Measured Gains (4x RTX 4090 `sm_89`, `8x512k @ 128k`, vs `v0.2.15` sync baseline)

- `SD_r512_c1`: `163.04 -> 176.24 tok/s` (+8.1%)
- `SD_r512_c8`: `653.91 -> 677.53 tok/s` (+3.6%)
- `SD_r65536_c4`: `393.91 -> 417.74 tok/s` (+6.0%)
- `SD_r65536_c8`: `608.71 -> 615.03 tok/s` (+1.0%)
- `MPD_a65536_4D_4P`: `ret=176.38% -> 188.93%` (+12.55 pp)
- `SD_r131072_c1`: `154.05 -> 159.69 tok/s` (+3.7%)
- `SD_r131072_c8`: `598.78 -> 602.08 tok/s` (+0.6%)
- `SD_r262144_c1`: `147.61 -> 150.69 tok/s` (+2.1%)
- `SD_r262144_c4`: `404.52 -> 409.35 tok/s` (+1.2%)
- `--smoke` (5k/slot, 4-slot): `100.0% / 0.0% / 100.0%`, `105.8 -> 125.0 tok/s` (+18.1%), `NEEDLE_GATE 8/8`, `leaks=0`, `0` spills, `0` Xids

### Wave K15 Measured Gains (4x RTX 4090 `sm_89`, `8x512k @ 128k`, vs Wave K14)

- **Opt-in controls:** `STRATA_KV_STAGE_SEL_GPU=1`, `STRATA_SAMPLER_ASYNC=1`, `STRATA_GDN_NORM_QFUSE=1`, `STRATA_GDN_CONV_GROUPS=1`, `STRATA_CONC_SUBCHUNK=512`, `STRATA_PREFILL_LOW_PRIO=1`, `STRATA_PARK_STAGE_POOL=1`, `STRATA_MLOCK_CODE=1`, `STRATA_DROP_STARTUP_MMAP=1`, plus group-spread `pick_slot`, zero-copy `memoryview` IPC, `PromptEncoder` singleflight, and `qsa_decode_attn` stack-spill elimination (`384–400 B -> 0–8 B`).
- `SD_r512_c1`: `176.24 -> 185.62 tok/s` (+5.3%)
- `SD_r512_c4`: `414.53 -> 444.40 tok/s` (+7.2%)
- `SD_r512_c8`: `677.53 -> 684.45 tok/s` (+1.0%)
- `SD_r65536_c1`: `149.87 -> 163.44 tok/s` (+9.1%)
- `SD_r131072_c1`: `159.69 -> 166.21 tok/s` (+4.1%)
- `SD_r131072_c4`: `382.69 -> 388.79 tok/s` (+1.6%)
- `SD_r262144_c8`: `637.71 -> 671.77 tok/s` (+5.3%)
- `MPD_a65536_4D_4P`: `dec_during_pf 283.15 -> 286.35 tok/s` (+1.1%)
- `max_expert_reads`: `50301.1 MB -> 0.0 MB`, `NEEDLE_GATE 8/8`, `leaks=0`, `new_xids=0`, 4-slot smoke `100.0% / 0.0% / 100.0%`.

### Wave K16 Measured Gains (4x RTX 4090 `sm_89`, `8x512k @ 208k` resident, vs Wave K15)

- **Opt-in controls:** `STRATA_PF_FP8_MIN_T=32` (default `128`; sub-128 `kBuckets[]` `16, 32, 48, 64, 96` in `src/prefill/gemm.cu`), plus `--kv-resident 212992` (`208k` resident KV, preserving `>= 600 MiB` free VRAM floor across all 4 GPUs).
- **8-Agent Short-Burst (65k ctx, 6 turns, deltas 64..512):** `turns/min` `215.95 -> 234.08` (+8.4%), `ttft_p50` `1191.4 -> 718.21 ms` (-39.7%), `agg_decode` `178.23 -> 181.09 tok/s`, `needles=8/8`, `leaks=0`, `min_vram_free=858 MiB`.
- **8-Agent Short-Burst (192k ctx, 4 turns, deltas 64..512):** `turns/min` `123.34 -> 137.42` (+11.4%), `ttft_p50` `2164.54 -> 1912.92 ms` (-11.6%), `needles=8/8`, `leaks=0`, `min_vram_free=842 MiB`.
- **`512..262144` Fast Gate:** `SD_r512_c1` `185.62 -> 202.79 tok/s` (+9.3%), `SD_r512_c8` `684.45 -> 706.91 tok/s` (+3.3%), `NEEDLE_GATE 8/8`, `leaks=0`, `min_vram_free=642 MiB`, `new_xids=0`, `max_expert_reads=0.0 MB`.
- **4-Slot Smoke (5k/slot):** `100.0% / 0.0% / 100.0%`, `Mean TTFT 690.0 ms`, `0` spills.
