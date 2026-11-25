<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Strata-AGX (`sm_89` Ada Lovelace Fork) — Measured Performance & Quality

`MAHDI-AQ/Strata-AGX` tracks upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata) `v0.2.16` (engine `0.1.62`, zero-GGML) with opt-in NVIDIA Ada Lovelace (`sm_89`) multi-GPU, FP8 prefill, QSA, MTP, and `O(delta)` server optimizations. **Default upstream execution stays byte-identical** (`Suite A` `a1_run`: `KL = 0.00000`, `top1 = 100.00%`) when opt-in switches are unset.

**Measured Coordinate:** `4× RTX 4090` (`96 GB` VRAM, Ada `sm_89`) · `Qwen3.8-Flash-Next IQ3_S` · `512K` max context · `8` concurrent agent slots

### Head-to-Head: Vanilla Native (`v0.2.16 / 58ee32277`) vs Strata-AGX (`ab410e06`)

| Workload / Metric | Vanilla Native | Strata-AGX | Improvement |
| --- | --- | --- | --- |
| **8×65K Agentic Burst** (`burst65`) | `11.74` turns/min | **`246.72`** turns/min | **`+2,001.5%`** (`21.0×`) |
| ↳ Burst TTFT p50 / Decode / Cold Prefill | `34.90 s` / `9.71` / `12,366` | **`1.42 s`** / **`244.92`** / **`16,145`** tok/s | **`-95.9%`** TTFT · **`25.2×`** dec · **`+30.6%`** PF |
| **8×196K Agentic Burst** (`burst196`) | `3.88` turns/min | **`236.48`** turns/min | **`+5,994.8%`** (`60.9×`) |
| ↳ Burst TTFT p50 / p95 | `69.78 s` / `192.72 s` | **`1.59 s`** / **`1.81 s`** | **`-97.7%`** p50 · **`-99.1%`** p95 |
| ↳ Agg Decode / Cold Prefill | `2.87` / `13,016` tok/s | **`162.21`** / **`17,077`** tok/s | **`56.5×`** dec · **`+31.2%`** PF |
| **400K Deep Context Needle** (cold / warm) | N/A (`270K` context cap) | **`Exact`** (`33.32 s` / `0.49 s`) | `512K` session support |
| **Fleet Rotation** (`I-7`, `2×20×65K`) | `1,428.5` / `1,426.1 ms` | **`552.8`** / **`552.5 ms`** | **`-61.3%`** rotation latency |
| **262K Prompt Hydration** (`SD_r262144_c4`) | `6,702.8` tok/s | **`16,077.7`** tok/s | **`2.40×`** |
| **`agx_fast_gate`** (`c1` / `c4` / `c8` / `MPD`) | — | **`204.2`** / **`640.1`** / **`791.3`** / **`394.4`** tok/s | `8/8` needles · `0` leaks |
**Arm H (2026-11-26 prefill profile, current `a1156dd97`):** `burst65` cold prefill p50 `3779.64 ms` (`-5.97%`); `burst196` cold prefill p50 `10947.96 ms` (`-4.89%`) and warm TTFT p50 `1375.16 ms` (`-13.28%`); `8/8` needles and `0` Xids across `burst65`, `burst196` and `agx_fast_gate`.

### Quality Parity (`challenge-first`) & Architecture

- **Suite B (`13/13 PASS`):** `100.0%` self-determinism · `24/24` (`100.0%`) think-greedy · `98.6%` think-sampled (vs `97.2%` Vanilla) · `0` empty answers & `0` repetition loops (vs `2` & `2` Vanilla).
- **Suite C (`8/8 PASS`):** `13/16` executed coding with `0` compile/syntax errors (vs `12/16`) · `12/12` (`100.0%`) OpenAI tool calls · `23/23` (`100.0%`) multi-turn repo adherence · `36/36` (`100.0%`) long-context accuracy.
- **Isolation:** `8/8` needles passed across all `agx_fast_gate`, `burst65`, and `burst196` runs with `0` cross-session leaks.
- **Opt-In `sm_89` + `O(delta)` Architecture:** Surgical per-row FP8 prefill, SM89 MMVQ interleave & prefetch, long-context top-k selection, BF16 side-stream handoff, `O(delta)` rolling `slot_held_crc`, compact `array('I')` tokenizer pipe, and `orjson` SSE streaming — see [docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md](docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md) and [docs/SETTINGS.md](docs/SETTINGS.md).

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

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.16-windows-x64-setup.exe` · `Strata-0.2.16-linux-x86_64.AppImage` · `Strata-0.2.16-macos-arm64.dmg`</sub>

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
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
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

## Strata-AGX (Ada Lovelace `sm_89` Fork)

`MAHDI-AQ/Strata-AGX` `main` tracks upstream `Niko1221/Strata` `v0.2.16` (engine `0.1.62`, zero-GGML) with opt-in NVIDIA Ada Lovelace (`sm_89`) multi-GPU, FP8 prefill, QSA, MTP, and `O(delta)` server optimizations (default upstream execution stays byte-identical when opt-in switches are unset).

- Architecture, verified `1×` / `2×` / `4× sm_89` flag profiles, spill/parity gates, and wave progression: [docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md](docs/SM89-ARCHITECTURE-AND-UPSTREAM-SYNC.md)
- Complete opt-in environment switch reference: [docs/SETTINGS.md](docs/SETTINGS.md)

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (generated from the public GitHub sponsors: one row per level, the biggest supporters first and largest; private sponsors are not listed) -->
<a href="https://github.com/codeconduit-official"><img src="https://images.weserv.nl/?url=github.com/codeconduit-official.png&w=240&h=240&fit=cover&mask=circle&output=png" width="120" height="120" alt="codeconduit-official" title="codeconduit-official"></a>

<a href="https://github.com/kod88vn"><img src="https://images.weserv.nl/?url=github.com/kod88vn.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/router0mail"><img src="https://images.weserv.nl/?url=github.com/router0mail.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="router0mail" title="router0mail"></a>
<a href="https://github.com/kingdonb"><img src="https://images.weserv.nl/?url=github.com/kingdonb.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kingdonb" title="kingdonb"></a>
<a href="https://github.com/Daisame"><img src="https://images.weserv.nl/?url=github.com/Daisame.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="Daisame" title="Daisame"></a>

<a href="https://github.com/ryan4yin"><img src="https://images.weserv.nl/?url=github.com/ryan4yin.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://images.weserv.nl/?url=github.com/themactep.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://images.weserv.nl/?url=github.com/macintog.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://images.weserv.nl/?url=github.com/bisonbet.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://images.weserv.nl/?url=github.com/vizv.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="vizv" title="vizv"></a>
<a href="https://github.com/seoulworker"><img src="https://images.weserv.nl/?url=github.com/seoulworker.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jaceknoga"><img src="https://images.weserv.nl/?url=github.com/jaceknoga.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jaceknoga" title="jaceknoga"></a>
<a href="https://github.com/jackyini"><img src="https://images.weserv.nl/?url=github.com/jackyini.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/moruku36"><img src="https://images.weserv.nl/?url=github.com/moruku36.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="moruku36" title="moruku36"></a>
<a href="https://github.com/hrkgk"><img src="https://images.weserv.nl/?url=github.com/hrkgk.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="hrkgk" title="hrkgk"></a>
<!-- supporters:end -->

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

