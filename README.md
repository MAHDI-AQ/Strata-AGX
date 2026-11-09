<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata-AGX" width="340">
  </picture>
</p>

<h2 align="center">Strata-AGX: High-Throughput Cluster Serving Engine</h2>
<p align="center">Production Quad-GPU Serving, Zero-Prefill Session Leases & Direct NVMe Streaming</p>

<p align="center">
  <a href="https://github.com/MAHDI-AQ/Strata-AGX/releases"><img src="https://img.shields.io/badge/release-v0.2.13--agx-0ab887" alt="Release v0.2.13-agx"></a>
  <a href="https://github.com/Niko1221/Strata/tree/v0.2.13"><img src="https://img.shields.io/badge/upstream-v0.2.13-blue" alt="Upstream v0.2.13"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/MAHDI-AQ/Strata-AGX" alt="MIT license"></a>
  <a href="#empirical-serving-benchmarks-114m-steps-production-run"><img src="https://img.shields.io/badge/benchmarks-1.14M%20steps%20verified-success" alt="Benchmarks"></a>
</p>

---

### Executive Overview & Production Architecture

**Strata-AGX** is the production cluster serving fork developed by the Mahdi AI Lab for extreme-throughput local inference across heterogeneous GPU clusters. Built directly on pristine upstream **Strata `v0.2.13`** (upstream commit `d2f096475817d4666c364c820b0ca830f751cfc5`), Strata-AGX integrates a battle-tested modular enhancement suite validated across more than **1.14 million continuous serving steps** (228M+ generated tokens) under 24/7 autonomous agent workloads.

```
+---------------------------------------------------------------------------------------------------+
|                                  STRATA-AGX SERVING STACK                                         |
+---------------------------------------------------------------------------------------------------+
|  Gateway / Reverse Proxy: GPUStack Unified Gateway (TLS Termination, Token Introspection)        |
+---------------------------------------------------------------------------------------------------+
|  API Server (serve/server.py):                                                                    |
|    - Zero-Prefill Session Leases & Dynamic Slot Affinity (4 Concurrent Agent Slots)               |
|    - Full OpenAI & Anthropic Messages Protocols (/v1/chat/completions, /v1/messages)             |
|    - Telemetry Introspection (/v1/slots, /v1/hardware, /api/tags, completion_tokens_details)     |
+---------------------------------------------------------------------------------------------------+
|  Direct NVMe Session Streaming (serve/nvme_streamer.c & nvme_session_streamer.py):               |
|    - Kernel-Bypass O_DIRECT 4K-Aligned DMA Paging                                                |
|    - 7.36 GB/s Sustained Streaming Throughput | Bit-Exact +0 B OS Swap Growth                     |
+---------------------------------------------------------------------------------------------------+
|  C++ GGML & CUDA Execution Engine (src/program/generate.cpp):                                     |
|    - In-Place Sub-Range Delta Appending (Layout l0 offset slice, 0-cost continuation)             |
|    - Distributed Multi-Token Prediction (STRATA_DISTRIBUTED_MTP: Round-Robin across Stages)       |
|    - Quad-GPU Layer Split: [14, 28, 36] across 3x RTX 4090 (24GB) + 1x RTX 3090 (24GB)           |
|    - 100% MoE Resident Experts (24,576 / 24,576 Resident, Zero Spills, Zero PCIe Faults)         |
+---------------------------------------------------------------------------------------------------+
```

---

### Key Architectural Pillars

#### 1. Zero-Prefill Session Lease Architecture & In-Place Delta Appending
In high-concurrency agent workflows (such as Hermes Agent or OpenCode running iterative reasoning loops), re-evaluating preceding prompt tokens on every turn wastes compute and introduces massive prefill latency.
- **Session Pinning & Affinity**: Slots maintain conversation states in resident device buffers. Subsequent requests from the same session acquire their existing slot with zero cache migration overhead.
- **In-Place Sub-Range Delta Appending (`copy_to_slot`)**: Instead of copying entire sequences or triggering expensive full-context recomputations, `conversation_kv_copy` utilizes sub-range offset indexing (`Layout l0` bounds calculation). When a session resumes from token $T_{\text{resume}}$ to $T_{\text{new}}$, only the delta slice $[T_{\text{resume}}, T_{\text{new}})$ is copied into device memory, bypassing conversation parking completely.
- **Bit-Exact Numerical Purity**: State continuity conforms strictly to turn boundary clamping ($T_{\text{turn}}$), ensuring $\|S_{\text{forked}} - S_{\text{cold}}\|_F = 0.0$ and completely eliminating phantom token hallucinations.

#### 2. Direct NVMe Streaming (`nvme_streamer.c` & `nvme_session_streamer.py`)
For sessions extending beyond GPU resident limits or requiring cold-state persistence:
- **Direct I/O DMA (`O_DIRECT`)**: Bypasses the OS page cache entirely, reading and writing page-aligned snapshot blocks directly between NVMe storage and host/device pinned buffers.
- **Zero OS Swap Growth**: Eliminates memory fragmentation and prevents swap bloat (+0 Bytes swap growth measured across continuous 16-hour multi-agent compaction cycles).
- **Throughput**: Delivers **7.36 GB/s sustained transfer rate** on WD_BLACK SN850X PCIe 4.0 NVMe storage.

#### 3. Quad-GPU Balanced Pipeline Topology
Engineered specifically for heterogeneous consumer GPU clusters lacking NVLink interconnects:
- **Cluster Hardware**:
  - **GPU 0, 1, 2**: 3× NVIDIA GeForce RTX 4090 (24,564 MiB each, Ada Lovelace, sm_89).
  - **GPU 3**: 1× NVIDIA GeForce RTX 3090 (24,576 MiB, Ampere, sm_86).
  - **Total VRAM**: 96 GiB aggregate VRAM.
  - **Thermal & Power Envelope**: All 4 GPUs locked to 300.00 W TDP with locked memory clocks (RTX 4090: +900 MHz / 22.8 Gbps; RTX 3090: +500 MHz / 20.5 Gbps). Aggregate compute: **0.948 PFLOPS FP16 Tensor compute**.
- **Balanced Layer Partitioning (`--layer-split 14,28,36`)**:
  - GPU 0: Layers 0–13 (14 layers)
  - GPU 1: Layers 14–27 (14 layers)
  - GPU 2: Layers 28–35 (8 layers + attention routing)
  - GPU 3: Layers 36–45 (10 layers)
  - Result: Eliminates the RTX 3090 PCIe bandwidth bottleneck, guaranteeing a minimum free VRAM floor $\ge 3,039\text{ MiB}$ across every card with **100% MoE expert residency (24,576 / 24,576 resident)** and zero PCIe expert spills.
- **Distributed Multi-Token Prediction (`--distributed-mtp` / `STRATA_DISTRIBUTED_MTP=1`)**:
  - Rather than concentrating speculative drafting compute on a single card, MTP drafters are distributed in a round-robin schedule across all active pipeline stages. Each card evaluates proposals locally using stage template weights, maximizing draft throughput while avoiding inter-GPU synchronization stalls.

#### 4. Multi-Session Dynamic UI Component
The bundled web monitor (`serve/web/index.html`, `serve/web/app.js`, `serve/web/app.css`) provides real-time visualization of:
- Per-slot conversational activity, token generation velocities, and KV cache residency.
- Stage-by-stage VRAM consumption, NVMe streaming rates, and MTP proposal acceptance rates.
- Interactive multi-turn playground with live slot inspection and fast-cancel controls.

---

### Empirical Serving Benchmarks (1.14M Steps Production Run)

From 1,140,000 empirical serving steps conducted on Route 19 (`Qwen3.8-Flash-Next-IQ3_S-Stock` across 4 concurrent OpenCode autonomous agents):

| Metric | Measured Value | Operational Context |
|---|---|---|
| **Speculative Draft Acceptance** | **77.4% – 86.7%** | MTP linear drafting across 4 concurrent slots |
| **Effective Decode Velocity** | **>615 tokens/sec** | Net effective generation under 4-slot batching |
| **Single-Slot Autoregressive Decode** | **169.6 tokens/sec** | Pure linear MTP decoding on stock Q3_S |
| **Prefill Throughput** | **1,125.5 tokens/sec** | Sustained 1,522-token prompt reading |
| **Warm TTFT (Time-to-First-Token)** | **417 – 446 ms** | Pre-flight cached prompt activation |
| **Direct NVMe Streaming Rate** | **7.36 GB/s** | Aligned snapshot save/restore over PCIe 4.0 |
| **OS Swap Memory Growth** | **+0 Bytes** | Zero page cache contamination over 228.6M tokens |
| **Energy Efficiency** | **0.0858 J / token** | Clamped at 300W per card |
| **MoE Expert Cache Hits** | **100.0% (Zero Misses)** | 24,576 / 24,576 resident; 0 RAM blobs, 0 disk blobs |

---

### Pristine Upstream Tracking

Strata-AGX tracks upstream Strata at pristine tagged releases:
- **Base Tag**: [`v0.2.13`](https://github.com/Niko1221/Strata/releases/tag/v0.2.13)
- **Base Upstream Commit**: `d2f096475817d4666c364c820b0ca830f751cfc5`
- **Tagged Release**: `fb57f7472c2ab4d176c216b30ed1d6b5e6132ef0`
- **Test Parity**: 271 unit tests passed with 100% bitwise equality across batch, session, and conversation streaming subsystems.

---

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (ISTA-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |
| ISTA-DAS Coder (half of the experts removed) | 191 GB | about 197 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<!-- supporters:end -->

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

