---
title: Phase 3 kernels - all three levers rejected, baseline retained
truth_class: evidence
project: local-agent-lab
updated: 2026-11-28
status: "CLOSED - no Phase 3 lever survived its gate; source reverted to a315e6e0a"
---

# Phase 3 — High-ROI CUDA prefill & MTP kernels (final outcome)

Branch `lever/phase3-kernels` started at `bae0172bf` off `main` = `a315e6e0a`.
Raw artifacts: `/srv/lab/scratch/2026-11-27-phase3/`.

| Lever | Intended effect | Gate result | Disposition |
|---|---|---|---|
| `STRATA_PF_FUSED_SMEM_LB=1` | NT=256 fused expert kernel without 40 B stack spill | Offline parity 4/4 and STACK 40->0, but **online first request kills the engine (exit 1, HTTP 503)** | Reverted |
| `STRATA_PIPED_SPEC_COST=1` | Cost-aware DraftPolicy on `--batch-mtp` slots | Candidate arm also 503; no measurement possible | Reverted |
| `STRATA_PROMPT_ATTN_Q4_IMMA=1` | Q4_0 INT8-MMA prompt attention | `p.v` second 32-dim block fails FP64 parity (2 failures) | Reverted before commit |

The first two were carried in `bae0172bf` as opt-ins pending an online A/B; that gate ran on
2026-11-28 and failed. The third was attempted in a later lane, failed its parity gate, and was
reverted before any commit. Full evidence:
[[2026-11-28-phase3-levers-rejected]] and [[2026-11-28-phase3-q4-imma-rejected]].

## What was proven before rejection

- L1: `cuobjdump -res-usage`, sm_89 `T_IQ2_S (22), GU=1, WW=2, NT=256, F16=0`: baseline 128 regs /
  STACK 40 -> opt-in 190 regs / STACK 0 / local 0; sm_86 likewise. `native_expert_parity` 4/4 on
  `iq2_s/{q2_0,iq4_nl,q5_0,q8_0}`; `check_cuda_spills.py` 0 not-allowed.
- L3: `q.k` and logits matched FP64 to 6 decimals after three fixes, but `p.v` failed the second
  32-dim block; a 1.17-1.20x synthetic timing did not excuse a parity failure.

## Final state

Source, settings register and `docs/SETTINGS.md` (manual Serve-side image reader footer preserved)
are byte-identical to `a315e6e0a`; a Release rebuild (CUDA 12.4.131, arch `86;89`) succeeds with
zero Phase 3 flag strings (build output is not byte-deterministic across clean rebuilds). The
served binary stays the already-qualified baseline
`4573e5df68895b99d9d78fea37ff4bf30e3f276eb03f721b5317c09cb744ce16`. The recipe stays at
`d97f39c223b0a48f88dac349d4bc9381650844317a12c21383d9703490494cc8`, `strata-serve` active,
`NRestarts=0`, 0 Xids. No rollback artifact was deleted.

The served configuration therefore remains the fully qualified Phase 1+2 baseline; the final
qualification battery on that baseline is recorded separately.
