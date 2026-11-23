## Strata-AGX (`v0.2.16` / engine `0.1.62` + SM89 Fork) — Mandatory Agent Rules

- **Ref Invariant (`HEAD == main == green`):** Keep `main` and the `green` tag aligned at the latest verified commit. Branch `feat/*`, `perf/*`, or `lever/*` from `main`, and promote back only after passing all spill, parity, docs, and serving isolation gates.
- **SM89 Pipeline & Session Lease Architecture:** Target Ada Lovelace (`sm_89`) cards with `--kv q4_0` (or `q8_0`), `--batch-mtp-mode shared --group-spread-mtp --concurrent-prefill`, `--resident-experts-only`, `--ple-io direct --ple-row-cache 16777216`, `--mtp-q4 all --mtp-window 32768`, and `STRATA_EMBED_VRAM=0` (`NativeEmbed::ensure_device` retained defensively; `--vanilla-defaults` restores upstream defaults). Radix trees, paged KV, HiCache, and SGLang-style prefix sharing are excluded in favor of **Session Lease + in-place delta appending + `O_DIRECT` NVMe streaming**.
- **CUDA Build & 3-Archive Spill Gate:** Configure from repository root with `cmake -S engine -B build -DCMAKE_CUDA_ARCHITECTURES="86;89"` (on Linux hosts with newer glibc, build with CUDA 12.4 + `g++-13` because CUDA 13.1 `crt/math_functions.h` `rsqrtf` conflicts with glibc). Run `python3 tools/check_cuda_spills.py build/libstrata_kernels.a build/libstrata_core.a build/libstrata_engine.a` and `python3 -m unittest tools.test_check_cuda_spills` — must report `0` disallowed spills across all `sm_89` targets and `OK`.
- **Verification & Docs Guards:** Pass idle-GPU parity (`batch_groups_test`, `batch_read_test`, `batch_rows_parity`, `kv_window_parity`, `gemm_fp8_parity`, `sampler_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity`) + `compute-sanitizer --tool memcheck`, `python3 tools/check --only docs`, `python3 -m unittest tools.test_settings_register`, and `pytest terminal/serve/test_server.py`. After running `python3 tools/settings_register.py --docs`, restore the hand-written `## Serve-side image reader switches (Terminal)` footer in `docs/SETTINGS.md` (the generator rewrites the full file).
- **Upstream Sync Audit:** Inspect divergence with `git fetch upstream && git log HEAD..upstream/main --oneline` and `python3 terminal/tools/upstream_sync_audit.py --repo . --upstream-ref upstream/main --fetch`, then re-run `python3 tools/check --only docs`, spill checks, and parity binaries before integrating upstream `Niko1221/Strata` updates.

---

# AGENTS.md

Version 0.2.16, engine 0.1.62. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.16-windows-x64-setup.exe`, `Strata-0.2.16-linux-x86_64.AppImage`, `Strata-0.2.16-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

- **`STRATA_READ_WIN_MAX_T=0` on multi-stage `--layer-split`:** Keep `STRATA_READ_WIN_MAX_T=0` (upstream default) when `ver.max_t() > S` (e.g., `--lookup-chain 3` with `--spec 4`) on multi-stage `--layer-split`; setting `1` allows `read_windows` to emit `T > S` (such as `T=6` on `enable_thinking=false` turn headers), which leaves the hidden state misaligned at `turn_at`.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (`python -m pytest terminal/tests/terminal`). The app's UI
  tests run with `npm test` in `app/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Wave K14–K26 Phase 1 Opt-In Controls (`sm_89` Fork on Upstream `v0.2.16`)

All fork optimizations below are **opt-in** (`terminal/data/settings-register.json`, default OFF -> byte-identical to upstream). See [docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md](docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md) for architecture, profiles, and measurements.

- **Wave K14:** `STRATA_SAMPLER_BATCH_ROWS=1`, `STRATA_PARK_STAGE_THREADS=1`, `STRATA_TOPK_ACTIVE_SM89=1`, `STRATA_KV_STREAM_WIDE=1`.
- **Wave K15:** `STRATA_KV_STAGE_SEL_GPU=1`, `STRATA_SAMPLER_ASYNC=1`, `STRATA_GDN_NORM_QFUSE=1`, `STRATA_GDN_CONV_GROUPS=1`, `STRATA_CONC_SUBCHUNK=512`, `STRATA_PREFILL_LOW_PRIO=1`, `STRATA_PARK_STAGE_POOL=1`, `STRATA_MLOCK_CODE=1`, `STRATA_DROP_STARTUP_MMAP=1`, `STRATA_HTTP_KEEPALIVE`.
- **Wave K16:** `STRATA_PF_FP8_MIN_T=32` (default `128`; sub-128 bucket sizing `16, 32, 48, 64, 96` in `engine/src/prefill/gemm.cu`).
- **Wave K17:** `STRATA_CONV_JEMALLOC_PURGE=1`, `STRATA_CONV_NOHUGEPAGE=1`, `STRATA_MSG_PREFIX_CACHE=1` (`terminal/serve/server.py`), `STRATA_PF_FP8_PAD_SMALL_T=1`, `STRATA_PF_SMALL_T_RUN=1`, `STRATA_SLOT_COPY_STREAM_SYNC=1`, `STRATA_ASYNC_STAGE_WAIT=1`, plus `Verifier::record_window` CUDA-graph logit writeback fix.
- **Wave K18:** `STRATA_PF_SMALL_T_RESIDENT_STAGED=1`, `STRATA_PF_RESIDENT_SMALL_T=1`, `STRATA_PF_FP8_SMALL_T_REUSE=1`, `STRATA_READ_WIN_NO_HEAD=1`, `STRATA_SLOT_PROMPT_CACHE=4`, `STRATA_CONV_CACHE_MEM_FLOOR=1`, `STRATA_MSG_PREFIX_CACHE=2` (`orjson` prefix cache v2).
- **Wave K19:** `STRATA_QSA_IDX_FP16=1`, `STRATA_KV_HOST_LAZY=1`, `STRATA_KV_RESOLVE_BATCH=1`, `STRATA_IPC_DELTA=1`, `STRATA_ROPE_TABLE=1`, `STRATA_SESSION_HASH=1`, `STRATA_GDN_QKVZ=1`, `zlib.crc32` `BGEND` framing.
- **Waves K20a–K20c (`v0.2.16` / `engine 0.1.62` zero-GGML selective port):** `STRATA_KV_STAGE_LAZY=1`, `STRATA_PROMPT_ATTN_SPLIT_KV=1`, `STRATA_PICK_SLOT_GROUP_SPREAD=1`, `STRATA_THP_DISABLE=1`, `STRATA_DROP_GGUF_PAGECACHE=1`, `STRATA_ADMIT_FIFO=0`, `--mtp-q4 all`, `--kv-resident 294912` (`288k`).
- **Waves K21–K24 Reliability & Memory Switches:** `STRATA_VISION_LAZY=1` / `STRATA_VISION_IDLE_S=300` (`terminal/serve/server.py`, plus `last_image_at` refresh in tool-image `finally`), `STRATA_KV_HOST_TRIM_IDLE=0` (opt-in dead-slot trim; left OFF by default), eviction-park cold re-park fallback (`conversation_kv_prefix_holds`), pooled-indexer byte-width-aware retention (`68451d330`), and 2 MiB NVMe session header overflow guard (`89fb8339d`).
- **Wave K25b & CI (`742a2b882`..`a7c8138d2`):** `STRATA_ATTN_MERGE_V2=1`, `STRATA_LM_HEAD_FP8_FAST=1` (XOR-swizzled SMEM + warp-4 parallel argmax epilogue when `!gumbel_enabled`, `STACK:0 LOCAL:0`), greedy LPT shard balancing + path-filtered `Checks` CI (`28m34s -> 7m49s` full / `3m34s` path-filtered), and explicit `setup_intel.py` load in `terminal/tools/test_setup_sycl.py`.
- **Wave K26 Phase 1 (`3706c4d34`, `3be67f536`, `4f073259f`):**
  - `STRATA_MTP_Q4_SHARED=1` (`engine/src/core/mtp.cpp`): shares `dense4_` / `q4_off_` across shared batch slots `1..7` with `owns_weights_` destructor guard (`SD_r262144_c8 +5.95%`, `b65/b196 +9.61%/+9.38%`, eliminates `Q80Block` small MMVQ `-99.27%`).
  - `STRATA_QSA_ROWS_RUNS=1` (`engine/src/kernels/cuda/qsa_select.cu`): splits contiguous same-slot runs (`<= 4`) onto `block_scores_rows_kernel<1..4, FP16>` (`-24.6%..-39.7%` kernel speedup, eliminates `block_scores_multi_kernel`, QSA scoring `-33.7%` live).
  - `STRATA_ATTN_NOSPLIT_MIN_CHUNKS=64` (`engine/src/kernels/cuda/qsa_decode_attn.cu`): sets `SPLIT_HEAD=false` for `!rot_q && n_chunks >= min` (`24/24` bitwise parity `0` bit diffs, `-10.7%` on `32k` `Q4_0`, `SD_r262k_c4 +12.98%`, `b196 agg_decode +30.40%`).
  - Verified on `4× RTX 4090` (`sm_89`, `8×512k @ 294,912`, `--layer-split 14,27,39 --split-device 3,2,1 --prefill 4096 --ple-io direct --ple-row-cache 16777216`): `SD_r262144_c8 = 716.39 tok/s`, `SD_r32768_c8 = 769.46`, `MPD_a65536 = 372.06`, `burst65 = 259.95 tpm` (`228.09 agg_dec`), `burst196 = 254.33 tpm` (`180.95 agg_dec`), isolated `400k` cold+warm exact (`950 MiB` min free VRAM), `2× 20×65k` fleet rotations (`20/20`, `0` leaks), `100/0/100` smokes, `0` disallowed spills, and `3/3` CI green.
- **Rejected / Disabled (`KEEP 0` / `OFF`):** `STRATA_SLOT_READ`, `STRATA_KV_COPY_RESIDENT`, `STRATA_TOPK_SPLIT`, `STRATA_PREFILL_HELP`, `STRATA_ATTN_NOSPLIT_DEEP`, `--mtp-window 8192`, `STRATA_GU_SWIGLU`, `STRATA_PIPED_SUFFIX`, `STRATA_PF_W8_CACHE_MIB=256`.
