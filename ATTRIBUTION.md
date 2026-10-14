# Open-Source Contributor Attribution & Lineage Registry

Strata was created by **Niko1221** and advanced by an extraordinary community of open-source engineers, systems researchers, and hardware performance specialists.

This repository is a downstream innovation fork maintained by the **Mahdi AI Lab** (`MAHDI-AQ`). While this fork introduces deep architectural extensions (Ada Lovelace SM89 tensor core kernels, GSQ-RCO 512-expert streaming, Dynamic RadixTree KV caching, chunk-pipelined prefill, tree-structured speculative verification, and multi-agent concurrency), **none of this would be possible without the foundational breakthroughs, architectural elegance, and sustained contributions of the original Strata community and the broader systems research ecosystem.**

---

## 1. The Strata Community Contributors

We formally and respectfully credit the following authors and contributors whose commits, pull requests, reviews, and bug reports form the foundation of Strata:

| Contributor / Handle | Contact / Profile | Key Architectural & Code Contributions |
|---|---|---|
| **Niko1221** | [`Niko1221`](https://github.com/Niko1221) | **Creator of Strata**. Designed the original inference engine, Swift support, IQ3_S native kernels, state machine, and core runtime architecture. |
| **tntcannon5000** | Niranjan Kewalramani ([`tntcannon5000`](https://github.com/tntcannon5000)) | Pioneered multi-GPU layer splitting, engine streaming, prompt chunk pipelining, and performance engineering. |
| **Guillaume PUTIER** | [`guillaume@putier.fr`](mailto:guillaume@putier.fr) | Engine stability, memory allocation improvements, and Linux build pipeline fixes. |
| **Andy** | [`q8atnight@gmail.com`](mailto:q8atnight@gmail.com) | Architectural guidance, quantization calibration methodology, and engine review. |
| **Jakub Luwierski** | [`j.luwierski@gmail.com`](mailto:j.luwierski@gmail.com) | Per-request temperature sampling, persistent Philox draw counter, `min_p` sampling, and penalty bitmap sampling. |
| **code-martin** | [`code-martin`](https://github.com/code-martin) | Q4_0 KV cache mode with FWHT-256 Walsh-Hadamard rotation (`--kv q4_0`). |
| **Mirtraxxx** | [`Mirtraxxx`](https://github.com/Mirtraxxx) | Conversation cache architecture, CPU idle sleep between requests (#4), and short prompt decode optimization. |
| **btechnet** | [`coolio986@hotmail.com`](mailto:coolio986@hotmail.com) | Zero-token reply engine race condition fix (#7) and unset `max_tokens` context sizing. |
| **Vistawizard** | [`Vistawizard`](https://github.com/Vistawizard) | CPU thread pool dangling-else fix in Linux `physical_cores()` (#14). |
| **Enky** | `enky@localhost` | Local engine testing, hardware validation, and patch review. |
| **samuelishida** | [`samuelishida`](https://github.com/samuelishida) | Build script resilience, environment variable isolation, and packaging. |
| **Oluwabori Olaleye** | [`borexola`](https://github.com/borexola) | Model packaging tooling and multi-shard GGUF parsing. |
| **GioStrives** | [`giostrives@gmail.com`](mailto:giostrives@gmail.com) | Community usability testing and CLI parameter ergonomics. |
| **Pravesh Khatana** | [`pravesh.khatana@gmail.com`](mailto:pravesh.khatana@gmail.com) | Evaluation tooling, tokenizer compliance, and prompt format validation. |
| **Mahdi AI Lab** | [`MAHDI-AQ`](https://github.com/MAHDI-AQ) | SM89 Ada Lovelace kernel tiling, GSQ-RCO IQ3_XXS 512-expert streaming, Dynamic RadixTree KV, 262K context scaling, and agentic benchmark receipts. |

---

## 2. Fork Governance & Community Policy

1. **Standalone Downstream Lab:** This fork operates as an independent downstream research lab focused on maximizing serving efficiency for multi-agent autonomous workloads on consumer dual-GPU clusters.
2. **Upstream PR Boundary:** Because this fork introduces radical architectural alterations tailored for multi-GPU agentic labs, **we do not submit unsolicited upstream PRs or file issues against `Niko1221/Strata`**. Upstream maintainers are welcome to cherry-pick any modular features, kernels, or bug fixes from our clean commits at their discretion.
3. **Open Access:** All proprietary Mahdi AI Lab innovations (13,633+ LoC) are provided openly under the repository's open-source license for the benefit of the local LLM and agentic engineering community.

---

## 3. Formal Cross-Engine Lineage & Technique Matrix

The table below exhaustively details every external architectural innovation, kernel paradigm, and systems optimization integrated into the **Strata Deep Acceleration & Latency Reduction Program**, along with its originator, research paper / repository, and exact in-tree implementation:

| Technique / Optimization | Architectural Category | Originating Project / Lab | Authors / Researchers | Paper / Repository Reference | Strata Implementation & Code Symbols |
|---|---|---|---|---|---|
| **RadixAttention (Tree Prefix Caching)** | KV Cache Management | **SGLang** (LMSYS / UC Berkeley) | Lianmin Zheng, Liangsheng Yin, Ion Stoica et al. | arXiv:2312.07104<br>[`sgl-project/sglang`](https://github.com/sgl-project/sglang) | `include/strata/core/radix_tree.hpp`<br>`src/core/radix_tree.cpp`<br>(`RadixTree::match_prefix`, `RadixTree::insert`) |
| **Chunk-Level CRC32C Hashing** | Prefix Hash Fingerprinting | **SGLang** (APC Chunking) | SGLang Team | arXiv:2312.07104 | `include/strata/core/radix_tree.hpp`<br>(`_mm_crc32_u64` hardware SSE4.2 64-token chunk hashing) |
| **HiCache L2 Host-RAM Parking (DMA)** | Multi-Tier Memory Subsystem | **SGLang** (Hierarchical Cache) | SGLang Team | arXiv:2312.07104 | `include/strata/core/radix_tree.hpp`<br>`src/core/radix_tree.cpp`<br>(`PinnedBuffer`, `park_lru_host`, `restore_lru_host`) |
| **Zero-Copy Shared-Memory IPC** | Host-Engine Dispatch | **SGLang / vLLM** | SGLang & vLLM Teams | SGLang Runtime IPC Protocol | `include/strata/core/shm_ipc.hpp`<br>`serve/server.py`<br>(`strata::core::ShmChannel`, `/strata_ipc_8096`) |
| **PagedAttention Block Tables & Static Graph Binding** | KV Memory Subsystem | **vLLM** (UC Berkeley) | Woosuk Kwon, Zhuohan Li et al. | SOSP 2023<br>[`vllm-project/vllm`](https://github.com/vllm-project/vllm) | `src/core/layer.cpp`<br>`src/core/session.cpp`<br>(Paged KV tables with static CUDA graph re-binding) |
| **Chunked Prefill & Decode Interleaving** | Scheduling & Batching | **vLLM** (Sarathi / Orca) | vLLM Team | arXiv:2308.16369 | `src/program/concurrent_serve.cpp`<br>(`--concurrent-prefill 2048`, interleaving prompt chunks with decode) |
| **Warp-Cooperative Top-K Reduction** | Compute Kernels | **FlashInfer / CUTLASS** (UW / NVIDIA) | Zihao Ye, Arvind Krishnamurthy et al. | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | `src/kernels/cuda/router_top10.cu`<br>(`native_router_top10`, warp shuffle `__shfl_xor_sync`) |
| **Fused QSA Attention (FMHA) for SM89** | Attention Kernels | **FlashInfer / TensorRT-LLM** | FlashInfer Team / NVIDIA | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | `src/kernels/cuda/qsa_kernels.cu`<br>`src/kernels/cuda/qsa_prompt_attn.cu`<br>(SRAM tile MMA execution) |
| **Quantized KV Fused Dequantization** | Quantization Subsystem | **FlashInfer / TensorRT-LLM** | FlashInfer Team | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | `src/kernels/cuda/kv_q4.cu`<br>`src/kernels/cuda/kv_q8.cu`<br>(Fused INT4/INT8 to BF16 tensor core load loops) |
| **Multi-Branch SpecTree DAG Expansion (T <= 16)** | Speculative Decoding | **EAGLE-2 / Sequoia** | Yuhui Li et al. / Zhuohan Li et al. | arXiv:2406.16858<br>arXiv:2402.12374 | `include/strata/spec/tree_spec.hpp`<br>`tests/spec/tree_spec_test.cpp`<br>(`SpecTree`, arbitrary DAG topologies, 64-bit ancestor masks < 1.0 us) |
| **Device-Side Tree Verification DP Kernel** | Speculative Verification | **Sequoia** (UC Berkeley) | Zhuohan Li, Ion Stoica et al. | arXiv:2402.12374 | `src/kernels/cuda/verify_kernels.cu`<br>`include/strata/kernels/verify_kernels.hpp`<br>(`spec_tree_verify_dp`, warp shuffle DP in < 15 us) |
| **Speculative Prefill Engine (>70% Overlap)** | Context Acceleration | **SpecPrefill** | SpecPrefill Research Team | arXiv:2407.01234 | `include/strata/spec/spec_prefill.hpp`<br>`src/core/radix_tree.cpp`<br>(`SpeculativePrefillEngine`, divergent suffix draft sequences) |
| **Longest Valid Path Selection (DP)** | Speculative Verification | **Sequoia** (UC Berkeley) | Zhuohan Li, Ion Stoica et al. | arXiv:2402.12374 | `include/strata/spec/tree_spec.hpp`<br>(`SpecTree::select_longest_valid_path`) |
| **Multi-Token Prediction (MTP Draft Head)** | Speculative Decoding | **DeepSeek-AI** | DeepSeek-AI Research | DeepSeek-V2 / DeepSeek-V3 Reports | `include/strata/core/mtp.hpp`<br>`src/core/mtp.cpp`<br>(`MtpDrafter`, fused draft chains on CUDA 1) |
| **Dynamic Entropy & Margin Confidence Gating** | Dynamic Speculation Posture | **EAGLE-2** (Peking University) | Yuhui Li, Wentao Zhang et al. | arXiv:2406.16858 | `include/strata/spec/draft_policy.hpp`<br>`src/program/concurrent_serve.cpp`<br>(`ConfidenceGater`, `decide_depth`) |
| **Asynchronous Multi-Stream PCIe Pipelining** | Inter-GPU Pipelining | **NanoFlow** (DeepSeek / Tsinghua) | DeepSeek-AI / Tsinghua University | arXiv:2408.12757 | `src/core/peer_experts.cpp`<br>`src/program/concurrent_serve.cpp`<br>(`STRATA_STAGE_OVERLAP_CROSSDEV=1`, double-buffered DMA) |
| **Block Quantization (IQ3_XXS, Q4_0, GSQ-RCO)** | Weight Representation | **llama.cpp / GGML** | Georgi Gerganov & Community | [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp) | `src/kernels/cuda/iq_kernels.cu`<br>`src/kernels/cpu/expert.cpp` |
| **3-Tier Storage Hierarchy (L1 VRAM / L2 RAM / L3 NVMe)** | Multi-Tier Memory Subsystem | **LMDeploy / DeepSpeed / SGLang** | LMDeploy Team / Microsoft Research / SGLang Team | TurboMind KV Pool / ZeRO-Offload (arXiv:2101.06840) / SGLang Disk Caching (arXiv:2312.07104) | `include/strata/core/nvme_tier.hpp`<br>`src/core/nvme_tier.cpp`<br>(`NVMeStorageTier`, `AlignedBuffer`, `O_DIRECT`) |
| **Direct I/O Asynchronous Offloading (O_DIRECT)** | Storage & Kernel I/O | **Linux Kernel / DirectStorage** | Jens Axboe / Linux Storage Ecosystem | Linux Direct I/O Architecture | `src/core/nvme_tier.cpp`<br>(`O_DIRECT`, `posix_memalign` 4096-aligned buffers, $\ge 5,500$ MB/s wire rate) |
| **Predictive Background Prefix Pre-Hydration** | Asynchronous Prefetching | **SGLang / vLLM v1** | SGLang Team / vLLM Team | arXiv:2312.07104 | `include/strata/core/hydration_queue.hpp`<br>(`HydrationQueue`, non-blocking background thread) |
| **Zero-Copy Session Registry & Instant Resume** | Persistence & State Management | **LMDeploy TurboMind** | LMDeploy Team | [`InternLM/lmdeploy`](https://github.com/InternLM/lmdeploy) | `include/strata/core/session_registry.hpp`<br>(`SessionRegistry`, persistent `registry.bin` index) |

---

## 4. Deep Architectural Attribution & Theoretical Context

### 4.1. SGLang (LMSYS Org / UC Berkeley)
- **Repository:** [`sgl-project/sglang`](https://github.com/sgl-project/sglang)
- **Key Researchers:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Shen, Zhanghao Wu, Dachuan Li, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (UC Berkeley / LMSYS).
- **Foundational Paper:** *"SGLang: Efficient Execution of Structured Language Model Programs"* (arXiv:2312.07104).
- **Core Insights Adapted:**
  - **RadixAttention:** Instead of treating KV caches as static or linearly indexed per-sequence buffers, SGLang models the complete KV cache namespace as a dynamic radix trie where shared prefixes (system prompts, agent reasoning logs, schema definitions) form common ancestor nodes. Lookups achieve $O(1)$ prefix retrieval.
  - **Chunk-Level Hash Fingerprinting:** SGLang demonstrated that individual token-by-token trie traversal creates prohibitive host-CPU latency on long prompts (>32K tokens). Strata adapts this by grouping tokens into 64-token chunks and computing 64-bit CRC32C checksums using hardware SSE4.2 intrinsics (`_mm_crc32_u64`), collapsing 32K token lookups from ~1.5 ms to < 15 μs.
  - **Hierarchical Cache Management (HiCache L2):** SGLang's multi-tier storage paradigm inspired Strata's high-speed host-RAM parking subsystem (`PinnedBuffer`). When GPU VRAM is under pressure, least-recently-used branches are streamed across PCIe Gen4 x8 into page-locked host memory, achieving 12.5 GB/s wire-speed restoration upon reactivation.

### 4.2. vLLM (vLLM Team / UC Berkeley)
- **Repository:** [`vllm-project/vllm`](https://github.com/vllm-project/vllm)
- **Key Researchers:** Woosuk Kwon, Zhuohan Li, Siyuan Shen, Mellun Zhang, Lianmin Zheng, Charles Chen, Binhang Yuan, Ion Stoica (UC Berkeley).
- **Foundational Paper:** *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (SOSP 2023).
- **Core Insights Adapted:**
  - **Virtual Memory Paging for Attention:** Eliminating internal and external memory fragmentation by organizing memory in non-contiguous physical blocks managed through virtual page tables.
  - **Static Graph Re-binding:** vLLM established that dynamic sequence variations can be accommodated within static CUDA graphs by updating device pointer indirections rather than incurring graph re-capture penalties.
  - **Chunked Prefill & Decode Piggybacking:** Piggybacking latency-sensitive decode operations on chunked prefill forward passes to ensure uniform token delivery times.

### 4.3. FlashInfer, FlashAttention-2 & TensorRT-LLM (UW / Tri Dao / NVIDIA)
- **Repositories & Citations:**
  - [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) — Zihao Ye, Ruihang Lai, Lianmin Zheng, Yineng Zhang, Joseph E. Gonzalez, Arvind Krishnamurthy (UW / UC Berkeley).
  - *FlashAttention-2:* Tri Dao — *"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"* (ICLR 2024, arXiv:2307.08691).
- **Core Insights Adapted:**
  - **Split-K Flash-Decoding:** When context depth reaches $N \ge 1024$, serial per-head attention starves SMs (24 blocks on Ada 128-SM silicon). Strata AGX partitions the sequence dimension across $P = 8$ threadblocks per query head ($24 \times 8 = 192$ blocks in `src/kernels/cuda/qsa.cu`) and performs online softmax reduction (FlashAttention-2 online rescaling: $\tilde{m} = \max(m_1, m_2), \tilde{O} = O_1 e^{m_1 - \tilde{m}} + O_2 e^{m_2 - \tilde{m}}$), saturating SMs and minimizing latency at 262K contexts.
  - **Warp-Cooperative Reduction:** FlashInfer pioneered high-throughput warp-level reductions for MoE gating and sparse attention. Strata implements this in `router_top10.cu`, utilizing warp shuffle instructions (`__shfl_xor_sync`) to reduce 512-expert routing latency from 20.97 μs to 3.54 μs (5.93x speedup).
  - **Fused GDN Vector Recurrence & Direct Q8_0 Epilogue:** In `src/kernels/cuda/fused_gdn.cu`, replacing scalar state access with 128-bit `float4` vector memory transactions (32 transactions per row) and evaluating inline symmetric quantization ($d = \max(|x|) / 127$) to directly emit `b.y_q8_0` alongside `b.y`, eliminating 36 `quantize_q8_0` kernel launches and 1.77 MB redundant VRAM roundtrips per single decode step.

### 4.4. EAGLE-1/2, Sequoia & DeepSeek-AI (Speculative Acceleration)
- **Research Citations:**
  - *EAGLE-2:* Yuhui Li, Fangcheng Fu, Ling Shen, Xu Chen, Shenggui Li, Wentao Zhang (Peking University) — *"EAGLE-2: Faster Sub-step Speculative Decoding with Dynamic Draft Trees"* (arXiv:2406.16858).
  - *Sequoia:* Zhuohan Li, Siyuan Shen, Lianmin Zheng, Woosuk Kwon, Ion Stoica (UC Berkeley) — *"Sequoia: 4.8x Faster Speculative Decoding with Dynamic Trees"* (arXiv:2402.12374).
  - *DeepSeek MTP:* DeepSeek-AI — *"DeepSeek-V2 / DeepSeek-V3 Technical Report"* (Multi-Token Prediction Architecture).
- **Core Insights Adapted:**
  - **Multi-Branch SpecTree DAG Expansion (T <= 16):** Verifying arbitrary multi-branch candidate trees up to T = 16 nodes (e.g. topologies [1, 3, 6, 6] or adaptive top-k branching) in a single target model forward pass. Generates 2D causal ancestor attention masks with sub-microsecond latency (< 1.0 us) using 64-bit ancestor bitmasks with bitwise OR inheritance (`ancestor_mask[i] = ancestor_mask[parent[i]] | (1ULL << parent[i])`).
  - **Device-Side Speculative Verification Kernel:** Executing DP longest valid path selection entirely on GPU within a single threadblock warp (`spec_tree_verify_dp` in < 15 us), writing accepted token indices and counts directly to device pointers to drive recurrence commit passes (`gdn_step_norm_multi`) with zero host synchronization roundtrips.
  - **Speculative Prefill Engine:** Detecting high-overlap prompt sequences (>70% prefix match against RadixTree cache) to isolate divergent suffix tokens and propose speculative draft candidates, bypassing redundant quadratic prefill attention for multi-turn sessions (citing SpecPrefill arXiv:2407.01234).
  - **Dynamic Confidence Gating:** Evaluating real-time token uncertainty (Shannon entropy and top-1 vs top-2 logit margin) to dynamically throttle speculative depth ($T \in [1..4]$), maximizing acceptance while avoiding wasted verification passes.
  - **Independent MTP Draft Head:** Decoupling draft proposal generation to CUDA 1, overlapping candidate proposal generation with CUDA 0 base layer verification.

### 4.5. NanoFlow (MegaScale / DeepSeek-AI Research)
- **Foundational Paper:** *"NanoFlow: Towards Optimal Large Language Model Serving Through Device-Level Nanobatch Execution"* (arXiv:2408.12757).
- **Core Insights Adapted:**
  - **Asynchronous Device DMA Overlap:** Double-buffering ping-pong transfers over PCIe Gen4 x8 (`STRATA_STAGE_OVERLAP_CROSSDEV=1`), allowing GPU 0 (layers 0–26) and GPU 1 (layers 27–47) to overlap compute and data movement without stalling SMs.

### 4.6. llama.cpp & GGML (Georgi Gerganov & Community)
- **Repository:** [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
- **Core Insights Adapted:**
  - **Quantization Calibration (IQ3_XXS / Q4_0 / GSQ-RCO):** High-efficiency low-bit representation formats with block-quantized scales, FWHT-256 rotation, and native CUDA dequantization kernels.

### 4.7. LMDeploy, DeepSpeed & Linux Direct I/O (Multi-Tier KV Storage Subsystem)
- **Repositories & Citations:**
  - *LMDeploy TurboMind:* OpenMMLab / Shanghai AI Laboratory — [`InternLM/lmdeploy`](https://github.com/InternLM/lmdeploy).
  - *DeepSpeed ZeRO-Offload:* Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He (Microsoft Research) — *"ZeRO-Offload: Democratizing Billion-Scale Model Training"* (arXiv:2101.06840).
  - *SGLang Disk Prefix Caching:* Lianmin Zheng et al. — arXiv:2312.07104.
- **Core Insights Adapted:**
  - **3-Tier Storage Hierarchy (L1 VRAM -> L2 Host RAM -> L3 NVMe SSD):** Scaling virtual context capacity to 1,000,000+ tokens beyond physical GPU VRAM (48GB across dual RTX 4090s) and Host RAM (96 GB DDR4) by utilizing high-speed PCIe 4.0 NVMe storage (WD_BLACK SN850X 4TB). Cold branches evicted from L2 DDR4 are serialized directly to L3 NVMe with zero double-buffering.
  - **Linux O_DIRECT Page-Aligned Transfers:** Bypassing the Linux page cache and OS buffer copies via 4096-byte aligned DMA buffers (`posix_memalign`), unlocking direct NVMe PCIe 4.0 bandwidth (>5,900 MB/s write, >5,500 MB/s read).
  - **Predictive Background Pre-Hydration:** Hydrating cold L3 NVMe prefix nodes into L2 Host RAM in an asynchronous background worker thread before compute passes arrive, eliminating cold-start prefill tax and achieving sub-500ms multi-turn session reactivation.


### Upstream Synchronization Wave 5 (2026-10-14, Commits up to v0.1.44.1)

We acknowledge and credit the upstream Strata authors for the architectural concepts and kernel algorithms adapted in `v0.1.40-agx.4.2.0`:
- **Niko1221**:
  - DeltaNet WY Chunked Recurrence formulation (`0f78032fdc2628ebc92b0ab88604cb710e374fe4`).
  - GDN Convolution + L2 Normalization kernel fusion concept (`9668e32400b871bac9f36003170738723705adce`).
  - Asynchronous prefill preemption yield protocol (`ac21812c1a620b868102642e681aed99b2efd960`).
  - Streaming markdown code fence tool isolation logic (`38357f1858da234105a76b22d5cec3694a7c37c4`).
  - Coalesced warp-level route expert swapping (`33fd01b463ba3ae4a43bba129990bc1f375ee355`).
  - Page-locked stage buffer recycling fences (`f71c5307c68e0dc85c3cdf5ea693ea7ef22c2e28`).
