## Strata-AGX (`v0.2.15` + SM89 Fork) — Mandatory Agent Rules

- **Ref Invariant (`HEAD == main == green`):** Keep `main` and the `green` tag aligned at the latest verified commit. Branch `feat/*`, `perf/*`, or `lever/*` from `main`, and promote back only after passing all spill, parity, and serving gates.
- **SM89 Pipeline & Session Lease Architecture:** Target Ada Lovelace (`sm_89`) cards with `--kv q4_0` (or `q8_0`), `--batch-mtp-mode shared --group-spread-mtp --concurrent-prefill`, and `"STRATA_EMBED_VRAM": "0"` (`NativeEmbed::ensure_device` retained defensively; `--vanilla-defaults` restores upstream defaults). Radix trees, paged KV, HiCache, and SGLang-style prefix sharing are excluded in favor of **Session Lease + in-place delta appending + `O_DIRECT` NVMe streaming**.
- **CUDA Spill Gate:** Run `python3 tools/check_cuda_spills.py <build-dir>/libstrata_kernels.a <build-dir>/libstrata_core.a` and `python3 -m unittest tools.test_check_cuda_spills` — must report `0` disallowed spills across all `sm_89` objects (`sampler`, `native_mmvq`, `qsa_select`, `gdn_chunk`, `iq_kernels`, `gemm`, `kv_stream`, `verify_kernels`) and `OK`.
- **8/8 Idle-GPU Parity:** With no competing server holding VRAM, pass `batch_groups_test`, `batch_read_test`, `batch_rows_parity`, `kv_window_parity`, `gemm_fp8_parity`, `sampler_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity`.
- **Upstream Sync Audit:** Run `python3 tools/upstream_sync_audit.py --repo . --upstream-ref upstream/main --fetch` before integrating upstream `Niko1221/Strata` updates.

---

# AGENTS.md

Version 0.2.15, engine 0.1.60. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.15-windows-x64-setup.exe`, `Strata-0.2.15-linux-x86_64.AppImage`, `Strata-0.2.15-macos-arm64.dmg`.
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
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
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
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
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

## Wave K14 Opt-In Controls

- `STRATA_SAMPLER_BATCH_ROWS=1`, `STRATA_PARK_STAGE_THREADS=1`, `STRATA_TOPK_ACTIVE_SM89=1`, `STRATA_KV_STREAM_WIDE=1` (registered in `data/settings-register.json` as `opt-in`; default paths remain bitwise identical).

## Wave K15 Opt-In Controls

- `STRATA_KV_STAGE_SEL_GPU=1`, `STRATA_SAMPLER_ASYNC=1`, `STRATA_GDN_NORM_QFUSE=1`, `STRATA_GDN_CONV_GROUPS=1`, `STRATA_CONC_SUBCHUNK=512`, `STRATA_PREFILL_LOW_PRIO=1`, `STRATA_PARK_STAGE_POOL=1`, `STRATA_MLOCK_CODE=1`, `STRATA_DROP_STARTUP_MMAP=1`, `STRATA_HTTP_KEEPALIVE` (registered in `data/settings-register.json` as `opt-in`; default paths remain bitwise identical).

## Wave K16 Opt-In Controls

- `STRATA_PF_FP8_MIN_T=32` (default `128`; lowers the SM89 FP8 prefill GEMM token threshold with sub-128 bucket sizing `16, 32, 48, 64, 96` in `src/prefill/gemm.cu`, registered in `data/settings-register.json` as `opt-in`; default path remains bitwise identical).
- Production `8x512k` resident KV headroom: `--kv-resident 212992` (`208k`), maintaining `>= 600 MiB` free VRAM floor across all 4 GPUs (`min_vram_free=642 MiB` under `8x262k` load).

## Wave K17 Opt-In Controls

- `STRATA_CONV_JEMALLOC_PURGE=1` (`include/strata/core/conversation_cache.hpp`: dynamic `mallctl("arenas.narenas")` + `arena.<n>.purge` fallback `arena.4096.purge` in `release_free_heap()`), `STRATA_CONV_NOHUGEPAGE=1` (`include/strata/core/conversation_buffer.hpp`: `MADV_NOHUGEPAGE` on `mmap` conversation segments).
- `STRATA_MSG_PREFIX_CACHE=1` (`serve/server.py`: `O(delta)` message-boundary prompt prefix cache respecting leading system merge, `tools`/`enable_thinking`/`reasoning_effort`, empty-assistant filtering, and non-`tool` checkpoint cuts).
- `STRATA_PF_FP8_PAD_SMALL_T=1` (`src/prefill/gemm.cu`: `T_pad >= 16` zero-padded FP8 prefill for `1 <= T < pf_fp8_min_t()` routed through `y_tmp_` + `T`-row FP32 scale/residual epilogue), `STRATA_PF_SMALL_T_RUN=1` (`src/prefill/prefill.cpp`: `run_impl` tail-block + `kv_stage_selected_from_host` for `T <= 256` above resident KV), `STRATA_SLOT_COPY_STREAM_SYNC=1` (`src/program/generate.cpp`: stream-scoped sync in `direct_slot_copy` / `copy_to_slot`), `STRATA_ASYNC_STAGE_WAIT=1` (`src/core/verify.cpp`, `src/core/mtp.cpp`: `cudaEventRecord`/`cudaStreamWaitEvent` for `copy_`, `commit_done_` on split stages, and async `draft_from_row` D2D staging).
- All 7 controls are registered in `data/settings-register.json` as `opt-in` (default paths remain bitwise identical).
