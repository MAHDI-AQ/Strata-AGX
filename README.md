# Strata-AGX

[![Target](https://img.shields.io/badge/target-Ada_Lovelace_sm__89-76b900.svg)](#multi-gpu-sm89-pipeline-recipes)
[![CUDA Spills](https://img.shields.io/badge/cuda_spills-0_across_sm__89_objects-brightgreen.svg)](#verification-spill-gate--parity-suites)
[![Upstream](https://img.shields.io/badge/upstream-Niko1221%2FStrata_v0.2.15-lightgrey.svg)](#upstream-attribution--sync)

**Ada Lovelace (`sm_89`) High-Throughput Multi-Slot Serving Fork of [`Niko1221/Strata`](https://github.com/Niko1221/Strata) (`v0.2.15`).**

Designed for **NVIDIA Ada Lovelace (`sm_89`)** GPUs — **RTX 4090**, **RTX 40-series**, and **L40S** in **1×, 2×, or 4× GPU** topologies — with zero-spill `sm_89` CUDA kernels, FP8 per-token prefill, Session Lease + `O_DIRECT` NVMe persistence, and full compatibility with standard Strata tooling and AI coding agents (**Claude Code**, **OpenCode**, **Codex CLI**, **Antigravity**, and **Hermes**).

---

## Standard Usage (`v0.2.15`)

Strata-AGX preserves the full upstream `v0.2.15` CLI, server, and MCP interfaces:

- **Terminal CLI (`strata launch` / `strata connect`):**
  ```bash
  # Preview launch plan without making changes
  strata launch --plan --family qwen --model iq3_s

  # Start OpenAI + Anthropic compatible server locally
  strata launch --family qwen --model iq3_s --yes --no-tui
  strata launch --status
  strata launch --show-key

  # Connect coding agents (Claude Code, OpenCode, Codex CLI, etc.)
  strata connect
  strata connect opencode --check
  ```
- **Python Server (`serve/server.py`) & Direct Engine (`build/strata --serve`):**
  ```bash
  # Direct engine server
  ./build/strata --serve --model /path/to/model.gguf --port 8080 --batch 8 --max-context 524288

  # Full OpenAI (/v1/chat/completions) & Anthropic (/v1/messages) API server
  python3 serve/server.py --model /path/to/model.gguf --port 8080
  ```
- **MCP Server (`tools/strata_mcp.py`):**
  Exposes model setup, launch, status, and connection management as MCP tools for **Claude Code**, **OpenCode**, **Codex CLI**, **Antigravity**, and **Hermes** (see `docs/MCP_SERVER.md`).

---

## Multi-GPU `sm_89` Pipeline Recipes

Strata-AGX supports homogeneous 1×, 2×, and 4× 24 GiB Ada Lovelace (`sm_89`) cards (and scales to 48 GiB L40S cards) using pipeline parallelism (`--layer-split`, `--split-device`), shared MTP drafting (`--batch-mtp-mode shared --group-spread-mtp`), and Q4_0 / Q8_0 KV caching.

### 4× 24 GiB Ada (`sm_89`, 96 GiB Total VRAM) — `--batch 8` & `--batch 4`

- **High-Concurrency 8-Slot (`8×512k @ 128k`):**
  ```bash
  ./build/strata --serve \
    --model /path/to/model.gguf \
    --batch 8 --max-context 524288 --kv-resident 131072 \
    --layer-split 13,26,39 --split-device 3,2,1 \
    --batch-mtp-mode shared --group-spread-mtp \
    --concurrent-prefill --prefill 4096 --kv q4_0
  ```
  *(With `STRATA_EMBED_VRAM=0` for maximum VRAM headroom across 4× 24 GiB cards.)*

- **High-Context 4-Slot (`4×1m @ 128k`):**
  ```bash
  ./build/strata --serve \
    --model /path/to/model.gguf \
    --batch 4 --max-context 1048576 --kv-resident 131072 \
    --layer-split 13,26,39 --split-device 3,2,1 \
    --batch-mtp-mode shared --group-spread-mtp \
    --concurrent-prefill --prefill 4096 --kv q4_0
  ```

### 2× 24 GiB Ada (`sm_89`, 48 GiB Total VRAM)

- **Balanced 4-Slot / 8-Slot:**
  ```bash
  ./build/strata --serve \
    --model /path/to/model.gguf \
    --batch 4 --max-context 262144 --kv-resident 65536 \
    --layer-split 24 --split-device 1,0 \
    --batch-mtp-mode shared --group-spread-mtp \
    --concurrent-prefill --prefill 4096 --kv q4_0
  ```
  *(Use `--batch 8 --max-context 131072 --kv-resident 32768` for higher concurrency at shorter context.)*

### 1× 24 GiB Ada (`sm_89`, 24 GiB VRAM)

- **Single-Card 4-Slot / 8-Slot:**
  ```bash
  ./build/strata --serve \
    --model /path/to/model.gguf \
    --batch 4 --max-context 131072 --kv-resident 32768 \
    --batch-mtp-mode shared --concurrent-prefill --prefill 2048 --kv q4_0
  ```
  *(Use `--batch 8 --max-context 65536 --kv-resident 16384` for 8-way concurrency on a single 24 GiB card.)*

---

## Wave K1–K14 `sm_89` Optimizations

Strata-AGX adds targeted Ada Lovelace (`sm_89`) optimizations while keeping `--vanilla-defaults` available to restore upstream defaults:

- **Zero-Spill `sm_89` CUDA Kernels (`src/kernels/cuda/`):**
  - `native_mmvq.cu`: `IlQ6K` routes `NC >= 7` (and `NC >= 5 && R > 1`) to `native_mmvq_il_kernel` (`STACK:0`) and `NC=5..6, R=1` to `native_mmvq_il_pf_kernel` (`DB=0, MINB=4`, `STACK:0`).
  - `sampler.cu`: Dead `double* ex` parameter removed in `sampled_tail_warp`; FP32 temperature clamp `p.temperature > 0.01f ? (1.0f / p.temperature) : 100.0f` (`0` spills).
  - `verify_kernels.cu`: `gdn_out_norm` 128-thread `float4` vectorized path (`0` spills).
- **FP8 Per-Token Prefill (`src/prefill/gemm.cu`, `src/kernels/cuda/qsa_select.cu`):**
  - Per-token FP8 E4M3 prefill quantization, pair-ticket activation (`STRATA_PF_FP8_PAIR_TICKET=1`), `STRATA_PREFILL_SMALL_T_RESIDENT=1` with tail-block-only `kv_stage_block_from_host`, `STRATA_QSA_HALVE_SM89=1`, and `STRATA_QSA_SLOT_RESOLVE=1`.
- **Session Lease + Delta Append + `O_DIRECT` NVMe Streaming (`serve/nvme_session_streamer.py`, `serve/server.py`):**
  - Radix trees, paged KV, HiCache, and prefix sharing are excluded in favor of deterministic **Session Lease + in-place delta appending + `O_DIRECT` NVMe streaming** (`STRATA_CONVERSATION_CACHE_DIR` configurable, `STRATA_CONV_NO_MLOCK=1` + `MADV_DONTNEED`, `STRATA_SKIP_SHORT_PARK_MAX=2048`).
- **L2 Persistence, Binary IPC, MTP Spread & Defensive Embed Placement:**
  - `STRATA_L2_PERSIST=1`: 49.50 MiB L2 persistence window with `pack_l2_persist` atomic pointer swap, 256 B alignment, and `~WeightTable` cleanup.
  - `STRATA_BINARY_IPC=1` (`BGENB`): Binary IPC framing between `serve/server.py` and `build/strata`.
  - `--group-spread-mtp`: Cross-GPU `pmtp_verdict` ordering with shared MTP drafter (`--batch-mtp-mode shared`) and parallel `ckpt_threads` `direct_slot_copy`.
  - `STRATA_EMBED_VRAM=0` + `NativeEmbed::ensure_device`: Mapped portable host memory + `init_native_embed_fp8` as the safe default (`NativeEmbed::ensure_device` retained defensively for multi-GPU safety).

---

## Verification, Spill Gate & Parity Suites

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

### CUDA Spill Gate (`tools/check_cuda_spills.py`)

Verify zero disallowed register spills or local-memory arrays across the built CUDA libraries:

```bash
python3 tools/check_cuda_spills.py build/libstrata_kernels.a build/libstrata_core.a
python3 -m unittest tools.test_check_cuda_spills
```

### Parity & Unit Suites

- **Idle-GPU Parity Binaries** (run with no competing server holding VRAM):
  `batch_groups_test`, `batch_read_test`, `batch_rows_parity`, `kv_window_parity`, `gemm_fp8_parity`, `sampler_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity`.
- **Terminal & Server Tests:**
  ```bash
  python3 -m pytest tests/terminal
  ```

---

## Upstream Attribution & Sync

Forked from [`Niko1221/Strata`](https://github.com/Niko1221/Strata) (`v0.2.15`). Audit divergence and dry-run merge compatibility against upstream with:

```bash
python3 tools/upstream_sync_audit.py --repo . --upstream-ref upstream/main --fetch
```

See [`docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md`](docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md) for the full `sm_89` architecture, Wave K1–K14 optimization map, and upstream sync protocol, and [`AGENTS.md`](AGENTS.md) for agent rules.
