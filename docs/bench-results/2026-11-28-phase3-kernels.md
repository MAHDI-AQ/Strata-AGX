---
title: Phase 3 kernels - partial delivery (two opt-in levers proven, Q4 IMMA and online battery pending)
truth_class: evidence
project: local-agent-lab
updated: 2026-11-28
status: "PARTIAL - 2 of 3 levers implemented and verified offline; NOT qualified for promotion"
---

# Phase 3 — High-ROI CUDA Prefill & MTP C++ kernels (partial)

Branch `lever/phase3-kernels` off `main` = `a315e6e0a223e967cb261801559d3e54123f3146`.
Raw artifacts: `/srv/lab/scratch/2026-11-27-phase3/` (offline logs, CSVs, build logs).

## Promoted first (board seq 31)

- `Strata-AGX`: main ff `41b26de66` -> `a315e6e0a`, `green` retagged, `lever/p2inc-items34` and `lever/badmit-stage2` deleted, `origin/main` pushed and verified by `git ls-remote`.
- `strata-opt-all`: fetched explicitly from `/srv/lab/Strata-AGX` (no configured remotes), main ff to `a315e6e0a`, `green` retagged.

## Lever 1 — `STRATA_PF_FUSED_SMEM_LB=1` (SHIPPED, offline-proven)

`engine/src/prefill/moe_fused_iq.cu`: added a sixth template parameter `LB2 = true` to `native_kernel`; `__launch_bounds__(NT, !LB2 || NT == THREADS ? 1 : 2)`. The NT=256 launch site selects the `LB2=false` instantiation only when the exact-`1` opt-in is set. Unset / `0` keeps the original instantiation and the default path byte-identical.

`cuobjdump -res-usage` / `check_cuda_spills.py --list`, sm_89, `T_IQ2_S (22), GU=1, WW=2, NT=256, F16=0`:

| instantiation | regs | STACK | local |
|---|---|---|---|
| `LB2=true` (default, baseline) | 128 | 40 | 0 |
| `LB2=false` (`STRATA_PF_FUSED_SMEM_LB=1`) | 190 | **0** | 0 |

sm_86 shows the same 40 -> 0 change. `native_expert_parity --synthetic <pair>` with `STRATA_PF_FUSED_NT=256 STRATA_PF_FUSED_SMEM_LB=1`: `iq2_s/q2_0`, `iq2_s/iq4_nl`, `iq2_s/q5_0`, `iq2_s/q8_0` = 4/4 `0 failures`, dequant bit-exact vs ggml. `check_cuda_spills.py --build-dir build` = 0 not-allowed on sm_86/sm_89.

UNVERIFIED: online burst65/196 A/B and end-to-end quality for this flag. The register spill is removed by construction, but no serving measurement was run in this lane.

## Lever 2 — `STRATA_PIPED_SPEC_COST=1` (IMPLEMENTED, ONLINE GATE PENDING — DO NOT PROMOTE)

`engine/src/program/generate.cpp`: new exact-`1` opt-in. Moves `bpolicy` above the chain lambda, and on `--batch-mtp` wires `bpolicy.set_context(sl.p)`, `bpolicy.keep_draft(...)` (instead of the solo policy), and `bpolicy.observe_probs(sl.dprob.data(), sl.n_drafts, accepted)` after the window verdict. Unset keeps `--spec-min-p`, the existing lookup policy's use of `bpolicy`, and the solo path byte-identically.

Build and offline gates pass, but the required gate — Suite D alpha plus interleaved A1/B/A2 on `burst65`/`burst196`, keeping only a neutral/positive result — has NOT been run. Per `strata-lever-measurement`, **this lever is not yet a candidate for promotion**. It is in the tree as an opt-in only so a fresh online lane can measure it; the next worker must either prove it neutral/positive or delete the code.

## Lever 3 — `STRATA_PROMPT_ATTN_Q4_IMMA=1` (NOT IMPLEMENTED)

No `prompt_attn_q4v3_kernel` was written. Porting the `imma16832` 24-bit split to Q4_0 needs two independent scale groups per 32-dim block on both products (q.k and p.v), doubling the accumulators and changing the shared-memory scale layout; it cannot be made bit-exact by a small wrapper. Shipping an unproven attention kernel would violate the parity/sanitizer law, so no stub or partial kernel was committed. This is a dedicated follow-up, not a refusal.

## Offline gates (all green)

- `cmake --build build --target strata native_expert_parity batch_admit_test -j8` RC=0 (Release, `CMAKE_CUDA_ARCHITECTURES=86;89`, g++-13, CUDA 12.4.131).
- `check_cuda_spills.py --build-dir build`: 0 not-allowed.
- `batch_admit_test`: all assertions passed.
- `tools.test_settings_register`: 8 tests OK. `capability_table.py --check`: ok. `tools/check --only docs`: 6 passed. `check_links.py --anchors`: 121 pages, all local links resolve.
- `PUBLIC_BOUNDARY_HITS=0`.
- Both new env vars registered in `terminal/data/settings-register.json` and `docs/SETTINGS.md` (manual `## Serve-side image reader switches (Terminal)` footer preserved).

## Not run in this lane

Full final qualification (Suites A-D, Challenge-20, EvalPlus, fast_gate, burst65/196, I-7, isolated 400k/smoke900, 7-Pillar), README showcase refresh with new verified numbers. No serving binary was replaced and no service restart was performed: `strata-serve` stayed active on cfg `d97f39c2`, bin `4573e5df`, 0 new Xids throughout.
