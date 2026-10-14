<h1 align="center">Strata AGX</h1>

<p align="center"><b>Multi-Agent Concurrency & Speculative Serving Fork for 125B MoE Models</b><br>
Dual NVIDIA GeForce RTX 4090 · Dynamic RadixTree KV · Continuous Multi-Agent Scheduling</p>

---

## Overview

**Strata AGX** is a downstream C++/CUDA serving fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) tailored for **multi-agent swarm concurrency**, **heterogeneous context scaling**, and **low-latency speculative decoding** on Dual NVIDIA GeForce RTX 4090 GPUs.

Upstream Strata demonstrated that 125B MoE architectures (like Qwen 3.8 Flash Next) can execute on dual consumer GPUs in single-stream mode ($C=1$). This fork focuses on the multi-agent serving constraints: eliminating FIFO queue head-of-line blocking, enabling zero-copy prefix sharing across parallel agent branches, and serving agent and tool traffic alike through one continuous-batching pool with a fair queue and prefix caching.

### Key Additions Over Upstream

1. **Continuous Multi-Agent Concurrency ($C=2$ on IQ3_XXS / up to $C=5$ on Q1 Coder):**
   - Replaced coarse round-based batching with token-level continuous micro-scheduling (Orca / vLLM v1 model).
   - Lockless dynamic slot recycling (sub-millisecond turnaround) upon stream completion.
   - Uniform agent slots at full context: $C$ concurrent streams at up to 262,144 tokens each; additional requests queue and are admitted as slots free (standard continuous-batching admission).
2. **Dynamic RadixTree KV Cache & 3-Tier Storage:**
   - Zero-copy prefix sharing across branching subagent sessions.
   - Sub-millisecond prefix forking for concurrent agent swarms.
   - 3-tier memory hierarchy: L1 VRAM (48GB) $\rightarrow$ L2 Host RAM (96 GB DDR4) $\rightarrow$ L3 NVMe DirectStorage (WD_BLACK SN850X @ 3,200+ MB/s Linux `O_DIRECT`).
3. **Speculative Decoding Engine (SpecTree DAG + Fused GDN):**
   - Multi-Branch SpecTree DAG expansion with 64-bit ancestor masks ($T \le 16$).
   - Fused GDN recurrence and Split-K online softmax attention, raising MTP acceptance yield from ~60% to **75%–90%**.
4. **Cross-GPU Prefill Pipelining (`STRATA_PREFILL_CHAIN=1`):**
   - Chunk-level stage pipelining across dual GPUs, achieving **3,450 tok/s** on IQ3_XXS (single-request path) and **5,355 tok/s** on Q1 Coder.
5. **Bitwise Parity Verification:**
   - Test harness confirming bit-for-bit parity against reference ggml across all 18 quant formats.

> [!NOTE]
> **Attribution & Upstream Boundary:**
> This repository is a standalone research fork. We credit **Niko1221** and the 15+ community contributors who created Strata. See [**`ATTRIBUTION.md`**](ATTRIBUTION.md) for full contributor credits. All commits are maintained modularly for upstream cherry-picking.

### Latest Line — `v0.1.40-agx.4.2.0` (Multi-Agent Concurrency & Hardware Acceleration)

On 2026-10-14, Strata AGX landed the **Multi-Agent Concurrency & Hardware Acceleration Release** (`v0.1.40-agx.4.2.0`, commit `7fdea49f`):

- **500ms Preemption Yield Hook (`BYIELD`, commit `2cc934a4`):** Integrated into the continuous multiplexed scheduler (`_generate_multiplex`). Slashes competing short-query TTFT from 7.59s to **2.35s** under heavy concurrent prefill bursts (upstream `ac21812c`).
- **CUDA-Native Fused GDN Conv + L2 Norm (`STRATA_GDN_CONVL2=1`, commit `f2921da6`):** Fuses 4-tap depthwise convolution and channel L2 normalization into a single CUDA kernel launch, eliminating **> 11 GB of dead DRAM writes per 16K prefill chunk** (upstream `9668e324`).
- **DeltaNet 32-Token Chunked Recurrence WY Kernel (`STRATA_GDN_CHUNKED=1`, commit `a22316a5`):** Forms $K \cdot K$ and $(I + A)^{-1}$ in 32-token chunks via WY representation across 128 SMs, speeding up GDN recurrence prefill by **1.8×** (upstream `0f78032f`).
- **Coalesced Warp Expert Swapping (commit `e5b97089`):** Replaces uncoalesced thread scans with 32-thread warp vote reductions (`__ballot_sync`), cutting router window latency by **7.8% (2.2ms)** (upstream `33fd01b4`).
- **Reasoning Code Fence Guard (commit `d79f294a`):** Strict markdown fence tracking protects reasoning mode (`</think>`) against accidental execution of quoted `<tool_call>` examples. Verified **3/3 (100%)** on tool fidelity tests (upstream `38357f18`).
- **Core Route H Architecture Clean-Up (commit `aa513381`):** Locks multiplexed asynchronous serving as canonical default and prunes legacy single-stream paths.
- **Zero-Copy Resident VRAM Routing (commit `7fdea49f`):** Preserves AGX proprietary resident routing (`m.gsrc_host`) across all 18,883 resident MoE experts.
- **Empirical Hardware Telemetry ($C=2$ on Dual RTX 4090s):**
  - **Streaming Decode:** **121.1 tok/s** (Stream 0) + **118.4 tok/s** (Stream 1) = **239.5 tok/s aggregate decode rate**.
  - **TTFT (Prefix Cache Hit):** **2.0 ms ($p50$)** on $T_{sys} / T_{turn}$ Radix-Tree boundaries.
  - **1-Hour Multi-Agent Soak:** 246,722 tokens generated across 5,747 turns with **0.00 MiB VRAM drift** and 0 media errors.

### Latest Line — `v0.1.40-agx.2.2.0` (Universal KV Caching Overhaul, Deep CoT Uncap & Stability Pack)

On 2026-10-10, Strata AGX landed the **Universal KV Caching Overhaul**, the **32k Reasoning Uncap**, and the multi-GPU kernel stabilization battery:

- **Universal KV Caching & LCP Partial Rollback:** Implemented intra-slot Longest Common Prefix ($K = \text{LCP}$) partial prefix rollback ($K \ge 256$), zero-copy truncating history and re-hydrating without re-prefill. Cross-session tool prompts (11k+ tokens) drop from 8.8s to **0.17s TTFT** (50x speedup), with full Tree-of-Thought rollback support.
- **L2 Host-RAM HiCache & L3 NVMe DirectStorage:** Inactive multi-session snapshot deadweight eliminated from VRAM (`STRATA_RADIX_VRAM_MIB: 0`), routing multi-session prefix caching to L2 Host-RAM (64 slots @ 10–12 GB/s physical PCIe 4.0 x8 DMA) and L3 NVMe (`O_DIRECT` @ 3.2 GB/s physical direct I/O). Freed 1.38 GiB of VRAM is reallocated into `--expert-cache auto` (+700 resident expert slots on CUDA 1).
- **Driver Lock Inversion & Deadlock Elimination:** Added strict bypass guards in `radix_tree.cpp` preventing unbudgeted device allocations and asynchronous `cudaFree` driver stalls. Enforced `admitted_this_round` exclusions across all passes (Pass 1, Pass 2, Pass 3, and Fallback), permanently resolving scheduler queue loop deadlocks.
- **Deep Reasoning Uncapped to 32k:** Raised default thinking ceiling to **32,768 tokens** and lowered reserve floor to 32 tokens, allowing complex multi-turn coding agents (Qwen 3.8 / DeepSeek) to reason through extensive chains-of-thought without premature cutoffs. Added `reasoning_budget_tokens` request override support.
- **Multi-GPU Pipelined Layer Split Hardening:** Enforced `cudaHostAllocMapped | cudaHostAllocPortable` across all unified group and expert pointer buffers, eliminating multi-GPU layer split memory access faults (`copy_i32: an illegal memory access` / Xid 31).
- **Dual-Slot Real-Time Monitoring & LAN Dashboard:** Model state card upgraded to two dedicated slot telemetry monitors (Slot 0 and Slot 1) tracking real-time status (`thinking`, `answering`, `tool_call`), exact context fill percentage, and per-slot decode throughput. Fully accessible across LAN at `http://<server-ip>:8096/monitor` and embedded in GPUStack.
- **Eager CSTOP & Sub-Threshold Sampler Clamping:** Reclamation of canceled requests on disconnect in under 500ms; sub-threshold temperatures (< 1e-4) clamped strictly to greedy argmax in both CUDA kernels and server frontend.

Engine identity: `0.1.40-agx.2.2.0` @ tag `v0.1.40-agx.2.2.0` (git tree verified).

### Previous Milestone — `v0.1.39-agx.2.0.0` (Upstream 0.1.39b Merged Baseline & Stability Pack)

Applied upstream merge of 0.1.39b performance line with:

- **Zero-Doorbell Multi-Token Verify Graph:** Eliminated host-driver submission bottlenecks during speculative verification; multi-token windows (1–4 tokens) captured into persistent CUDA execution graphs.
- **Sub-Warp Expert Packing & Shared-Mem Staging:** Replaced coarse expert launches with warp-level cooperative packing, raising warm decode from ~38 tok/s to **80–100+ tok/s** bit-exact across Dual RTX 4090s.
- **Claude Code Multi-Turn Cache Hit Pinned:** Cherry-pick `bae372b9` pins `x-anthropic-billing-header` per-request stamps, ensuring agent turns hit identical prefix hashes across consecutive prompts.
- **Literal Control Token Encoding:** Cherry-pick `98b7ea94` sanitizes literal control tokens (`<|im_start|>`, `<|im_end|>`) in user messages to prevent template desync.
- **High-Burst HTTP Backlog:** Cherry-pick `c34dd571` sets `STRATA_HTTP_BACKLOG 256` to absorb multi-agent burst connection attempts without TCP drop.
- **GPUStack Native Reservation & Tokenize Endpoint:** Restored `/v1/tokenize` endpoint with full token-granular radix reservation metadata for automated GPUStack admission.
- **Dual-GPU Real-Time Monitoring:** Upgraded the embedded monitor dashboard with dedicated per-card telemetry for CUDA 0 (layers 0–26, 12,077 MoE cache slots) and CUDA 1 (layers 27–47 + head, 8,662 MoE cache slots).
- **Auxiliary Slot Total Purge:** Aux-slot experiments and urgent preemption clamps permanently purged (`a3da023c`). All agent requests and tool calls pass through the single continuous queue with $C=2$ concurrent streams ($2 \times 262\text{k}$ KV pool).

Engine identity: `0.1.39-agx.1.2.0` @ tag `v0.1.39-agx.1.2.0` (git tree verified).

---

## Empirical Benchmark Verification

All measurements below were conducted directly on physical hardware running real multi-turn completions:

- **Rig:** Dual NVIDIA GeForce RTX 4090 24GB (PCIe 4.0 x8 / x8 bifurcation)
  - *Clocks:* Core Offset **+150 MHz** (~2,800 MHz boost) · GDDR6X Memory Offset **+1000 MHz** (11.5 GHz effective / ~1,104 GB/s per GPU)
- **CPU:** AMD Ryzen 9 5950X (16-Core / 32-Thread)
- **RAM:** 96 GB DDR4-3200 (32+16+32+16 dual-channel 4-DIMM layout)
- **Storage:** WD_BLACK SN850X 4TB NVMe SSD
- **OS:** Ubuntu 24.04 LTS

---

### 1. Single-Stream ($C=1$) Baseline — Qwen 3.8 Flash Next IQ3_XXS

Evaluated on `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ3_XXS, 125B MoE with 6B active parameters, 512 experts, 3.06 bpw) using `--max-context 262144` and `--kv q4_0`:

| Metric | Measured Baseline | Program v2 (`v0.1.38-agx.1.0.5`) | Delta |
|---|---:|---:|---:|
| **Single-Stream Decode (Short Context)** | 100.5 – 105.2 tok/s | **115.2 – 117.8 tok/s** *(8.49 – 8.68 ms/tok)* | **+9.5% to +14.6%** |
| **Single-Stream Decode (Medium Context, 2.8K)** | 105.2 tok/s | **107.6 tok/s** *(9.30 ms/tok)* | **+2.3%** |
| **Single-Stream Decode (Peak Observed)** | 108.6 tok/s | **129.2 tok/s** *(7.74 ms/tok)* | **+19.0%** |
| **MTP Speculative Acceptance Yield** | ~60.0% – 65.0% | **75.0% – 89.7%** | **+15.0% to +29.7%** |
| **Cold Prefill Throughput** | ~850 tok/s *(Unpipelined)* | **3,450.2 tok/s** *(`STRATA_PREFILL_CHAIN=1`, single-request path `--prefill auto`, 16,384-token chunks)* | **4.1× speedup** |
| **VRAM Expert Residency (262K Context)** | 86.3% (21,209 / 24,576 slots) | 86.3% (21,209 / 24,576 slots) | Preserved |

*Note: Upstream Strata natively supports 262,144 context in single-stream mode; Strata AGX accelerates this single-stream decode path via fused GDN recurrence and Split-K FMHA.*

*Note on the prefill row: the 3,450.2 tok/s figure is measured on the single-request path at `--prefill auto` (16,384-token chunks); the concurrent serving path utilizes `--concurrent-prefill 2048` (right-sized down from 5120/12288) to reclaim VRAM into the resident expert pool while keeping prompt chunk execution bounded.*

---

### 2. Multi-Agent Concurrency & Throughput Scaling

#### A. Homogeneous Multi-Agent Swarm — Qwen 3.8 Flash Next Coder Q1 (`IQ1_M`, 128K Context)
*100.0% resident experts in VRAM; zero PCIe host stalls:*

| Concurrency ($C$) | Serving Mode | Aggregate Throughput | Per-Agent Decode | Cold Prefill | VRAM Residency |
|---|---|---:|---:|---:|---:|
| **$C=1$** | Single Stream | **100.5 – 108.6 tok/s** | 100.5 – 108.6 tok/s | 2,380 tok/s (short) / 5,124 tok/s (bulk) | 100.0% (pinned) |
| **$C=2$** | Dual Agent | **152.40 tok/s** | 100.20 tok/s | — | 100.0% |
| **$C=3$** | Multi-Agent Swarm | **187.50 tok/s** | 82.10 tok/s | — | 100.0% |
| **$C=4$** | Swarm Fan-Out | **205.10 tok/s** | 68.40 tok/s | — | 100.0% |
| **$C=5$** | Full Concurrency (Burst) | **220.04 tok/s** | 59.4 – 68.7 tok/s | 5,355.4 tok/s | 100.0% |
| **$C=5$** | Sustained Sweet Spot (11.5 GHz) | **253.92 tok/s** | 59.2 – 63.2 tok/s | 5,124.2 tok/s | 100.0% |

#### B. Production Serving Posture — Qwen 3.8 Flash Next `IQ3_XXS` ($C=2$, Dual-Agent Concurrency)
*Measured on active production release `v0.1.40-agx.2.2.0` (Dual RTX 4090 24GB @ PCIe 4.0 x8, Ryzen 9 5950X, 96GB RAM):*

| Concurrency ($C$) | Serving Mode | Aggregate Throughput | Per-Slot Decode | Cold Prefill (62k) | Warm TTFT (Cached) | VRAM Expert Residency |
|---|---|---:|---:|---:|---:|---:|
| **$C=1$** | Dedicated Solo | **105.0 – 120.4 tok/s** | 105.0 – 120.4 tok/s | ~11.0 s (~5,631 tok/s) | 24 – 35 ms | 84.4% (20,739 / 24,576 slots) |
| **$C=2$** | Dual Continuous | **165.0 – 186.9 tok/s peak** *(82.1 – 98.4 sustained steady-state)* | 87.0 – 99.5 tok/s peak | ~19.7 s (~3,180 tok/s) | 24 – 35 ms | **70.2% (17,254 / 24,576 slots)** |

> [!NOTE]
> **Active Production Guarantee:** $C=2$ is the optimal operating sweet-spot on Dual 24GB RTX 4090s. It guarantees non-blocking multi-agent serving with dedicated 5k prefill workspaces while maintaining 70.2% resident MoE experts in VRAM, eliminating PCIe expert thrashing.

#### C. Retired Tiered Heterogeneous Swarm — Qwen 3.8 Flash Next `IQ3_XXS` (852K Virtual Context)
*Historical (`v0.1.38-agx.1.0.5`): the tiered auxiliary-slot experiment was removed from the engine on 2026-10-06 as a regression source (it collapsed prefill under load even with no aux traffic). Tool/aux traffic now flows through the standard pool + queue; these figures document the retired build.*
*Evaluated with 3 Primary slots @ 262K context + 1 Aux slot @ 65K context = **851,968 tokens context** (66.6% VRAM residency; 33.4% streamed dynamically from host DDR4 over PCIe 4.0 x8):*

| Stream / Role | Context Allocation | Measured Decode Speed | Step Latency | Turnaround / Wall Time | MTP Acceptance |
|---|---|---:|---:|---:|---:|
| **Main Orchestrator (Stream 0)** | 262,144 tokens | **58.5 tok/s** | 17.11 ms | 2.34 – 61.5 s | 88.8% |
| **Subagent 1 — Coder (Stream 1)** | 262,144 tokens | **36.6 – 40.3 tok/s** | 24.80 – 27.35 ms | 2.38 – 40.5 s | 87.5% |
| **Subagent 2 — Verifier (Stream 2)** | 262,144 tokens | **42.1 – 50.0 tok/s** | 20.02 – 23.73 ms | 2.45 – 61.4 s | 89.7% |
| **Auxiliary Tool Query (Stream 3)** | 65,536 tokens | **43.4 tok/s** | 23.02 ms | **2.07 s** *(Sub-15ms preemption)* | 91.2% |
| **Total Swarm Capacity** | **851,968 tokens** | **139.9 – 148.8 tok/s** *(Aggregate)* | — | **2.46 s** *(162 toks short fanout)* | **88.8% – 89.7%** |

---

### 3. Head-to-Head: Vanilla Upstream Strata (v0.1.38) vs Strata AGX (`v0.1.38-agx.1.0.5`)

Evaluated side-by-side on the exact same Dual RTX 4090 rig:

> *Note (2026-10-06): the auxiliary-slot machinery referenced in the rows below was removed from the engine as a regression source; concurrent serving is now a uniform N-slot pool with queue admission. Rows document `v0.1.38-agx.1.0.5`.*

| Capability / Benchmark Metric | Vanilla Upstream Strata (v0.1.38) | Strata AGX (`v0.1.38-agx.1.0.5`) | Empirical Difference |
|---|---|---|---|
| **Multi-Agent Scheduling** | **$C=1$ FIFO Serialization** *(Incoming requests wait for active generation to finish)* | **Continuous Micro-Scheduling ($C=2$ on IQ3_XXS / $C=5$ on Q1)** *(Iteration-level dynamic slot recycling)* | Concurrent multi-agent execution |
| **Auxiliary Request Turnaround** | >15.0 – 22.0 s *(HoL blocking behind active agent output)* | **2.07 s** *(Sub-15ms preemptive queue insertion)* | **86.2% latency reduction** for agent tool calls |
| **Total Concurrent Virtual Context** | 262,144 tokens *(Single slot only)* | **851,968 tokens** *(3× 262K primary + 1× 65K aux)* | **3.25× total concurrent active context** |
| **Prefix Caching & Tree Branching** | Static per-request buffer; cross-GPU parking unsupported | **Dynamic RadixTree** with zero-copy prefix sharing & L2/L3 parking | Zero redundant prefill tokens on agent conversation forks |
| **Aggregate Swarm Throughput ($C=3$)** | 77.48 tok/s *(Serialized FIFO queue)* | **187.50 tok/s** *(Parallel decoding)* | **2.42× throughput scaling** |
| **Aggregate Swarm Throughput ($C=5$)** | 81.73 tok/s *(Serialized FIFO queue)* | **220.04 tok/s (burst) / 253.92 tok/s (sustained)** | **3.11× throughput scaling** |
| **Speculative Acceptance Yield** | ~60% – 65% | **75.0% – 89.7%** | Fused GDN & 64-bit ancestor SpecTree DAG |
| **Storage Tier Hierarchy** | VRAM only | **3-Tier (L1 VRAM $\rightarrow$ L2 Host RAM $\rightarrow$ L3 NVMe DirectStorage)** | Persistent sessions via Linux `O_DIRECT` @ 3,200+ MB/s |

---

## Building & Installation

### Requirements
- **Operating System:** Linux (Ubuntu 22.04 / 24.04, Debian 12, or WSL2)
- **GPU:** Dual NVIDIA GeForce RTX 4090 / 3090 (24GB VRAM per GPU)
- **CUDA Toolkit:** CUDA 12.4+ (with `nvcc` and `g++-13`)
- **Build Tools:** CMake 3.24+, Ninja, Python 3.10+

### Compile from Source
```bash
git clone https://github.com/MAHDI-AQ/Strata-AGX.git
cd Strata-AGX
cmake -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON
cmake --build build -j 16
```

### Bitwise Parity Suite
Verify that all custom SM89 kernels and RadixTree implementations match reference outputs bit-for-bit:
```bash
./build/iq_multi_parity
./build/s2_expert_grouped_parity
./build/native_grouped_parity
./build/radix_tree_test
```

---

## Serving & API Usage

### Starting the Server
```bash
python3 -m serve.server --port 8096 --host 0.0.0.0
```

### Reference Concurrent-Serving Posture (Dual-4090 2 × 262K Slots)
```bash
python3 serve/server.py --engine strata --config configs/q3-xxs-dual-4090-serving.json --port 8096
# config highlights: 2 slots × 262,144-token context ($C=2$), --concurrent-prefill 2048,
# --kv q4_0, dynamic RadixTree L1/L2/L3 (VRAM → host RAM → NVMe)
```

### Client Example (Python / OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8096/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="strata-coder",
    messages=[
        {"role": "system", "content": "You are an autonomous coding assistant."},
        {"role": "user", "content": "Write a lockless triple-buffer circular ring index advancement in C++20."},
    ],
    temperature=0.6,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

---

## Modularity & Portability

- **Upstream Cleanliness:** All additions are structured by subsystem; cherry-picking into upstream Strata requires no architectural rewrites.
- **Zero Hardcoded Paths:** All paths and configurations are dynamic and relative; no local lab paths exist in defaults.
- **Graceful Fallbacks:** On single-GPU or non-Ada architectures, features fall back cleanly to standard linear execution paths.

---

## License & Attribution

Strata is licensed under the **Apache License 2.0 / MIT License**.
See [**`ATTRIBUTION.md`**](ATTRIBUTION.md) for contributor credits and lineage.
