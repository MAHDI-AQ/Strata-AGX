---
title: Phase 3 final qualification - retained baseline (Phase 1+2)
truth_class: evidence
project: local-agent-lab
updated: 2026-11-28
status: "PASS - all 13 gates ok; Phase 3 levers rejected; baseline binary/config retained"
---

# Final qualification battery — 2026-11-28

Candidate: the retained baseline `/usr/local/bin/strata` =
`4573e5df68895b99d9d78fea37ff4bf30e3f276eb03f721b5317c09cb744ce16`, recipe
`/srv/lab/recipes/stock-q3-s/config.json` =
`d97f39c223b0a48f88dac349d4bc9381650844317a12c21383d9703490494cc8`.
Hardware: `ai-server`, 4x RTX 4090 (sm_89), `Qwen3.8-Flash-Next IQ3_S`, 4 concurrent slots,
`AGX_GPU_POLL_S=20.0`, `AGX_ENABLE_DMON=0`, `nvtop` absent, **0 Xids** end to end.

All 13 gate verdict files are `ok: true`:
`/srv/lab/scratch/2026-11-27-phase3/final_qual/{online,offline}/*/verdict.json`.

## Speed / isolation (fresh, 2026-11-28)

| Probe | Result |
|---|---|
| `burst65` (8 slots, 3 turns, 65K) | `344.66` turns/min, wall `4.178 s`, warm p50 `644.30 ms`, p95 `953.35 ms`, `251.32` agg-dec tok/s, `8/8` needles, `0` leaks, min free VRAM `988 MiB`, `0` Xids |
| `burst196` (8 slots, 4 turns, 196K) | `233.88` turns/min, wall `8.209 s`, warm p50 `1038.25 ms`, p95 `1521.04 ms`, `190.88` agg-dec tok/s, `8/8` needles, `0` leaks, `988 MiB`, `0` Xids |
| `I-7` (`2x20x45K`, 8 slots, 2 rounds) | `all_ok=true` in both runs, `20/20` per round, `0` leaks, `p50 536.2 / 536.4 ms`, evictions `32 / 40`, parked `32 / 40`, `0` missing prefixes |
| isolated `400K` | `exact_match=true`, cold `26.97 s` (401,600 tokens, `15,307` prompt tok/s), warm `0.42 s`; min free VRAM `988 MiB` |
| `post400k_smoke` | `accuracy 100.0%`, `contamination 0.0%`, mean TTFT `547.2 ms`, `9,374.4` prefill tok/s, `171.1` decode tok/s |
| `agx_fast_gate --slots 8` | `8/8` needles, `0` leaks, rungs `512 / 32768`, `0` Xids |

## Quality / parity

- **Suite A** (service stopped, its own engine, 5 passes): `a1_run`, `a1_prompt`, `a2_small_t`,
  `a3_prompt`, `a3_tail` all `ok`, `nonfinite=0`.
  `compare-a` against the canonical challenge-first reference
  (`/srv/lab/scratch/2026-11-27-phase1/step3/offline/ref`) **PASS**: `KL_mean=0.00000` on every
  pass, `top1=100%`, `dPPL=+0.000%` for `a1_run` and `a3_tail`.
- **Suite B**: determinism `62.5%`, think-greedy `100.0%`, think-sampled `98.61%`, no-think
  `95.83%`, `0` empty answers, `0` repetition loops.
- **Suite C**: `16/16` executed coding, `12/12` tools, `23/23` repo turns, `36/36` long-context,
  `0` leaks/loops.
- **Suite D**: `8` cells, MTP alpha `0.5892–0.6708`, `0` request errors.
- **Challenge-20**: `18/20` (`90.0%`), curated `12/12`, new-8 `6/8`.
- **EvalPlus** (via pillar4): `12/12`.
- **7 pillars**: `P2 119/120` (20K–491K ladder, `0` leaks/loops), `P3` think-greedy `100%` /
  sampled `100%` / no-think `100%`, `P4 4/4` trajectories with `0` schema errors, `P6 8/8` recall
  and `256/256` delta steps with `0` leaks, `P7 8/8` needles and `0` errors.
- **Offline**: `native_expert_parity --synthetic iq2_s/{q2_0,iq4_nl,q5_0,q8_0}` —
  `0 of 4,915,200` values differ in any bit; `check_cuda_spills.py --build-dir build` —
  `0` not-allowed on sm_86/sm_89; `batch_admit_test` — all assertions passed;
  `tools.test_settings_register` 8/8; `capability_table.py --check` ok; `tools/check --only docs`
  6/6; `check_links.py --anchors` 121 pages resolve; `PUBLIC_BOUNDARY_HITS=0`;
  `bench_results_index.py --check` ok.

## Phase 3 disposition

No Phase 3 lever survived: `STRATA_PF_FUSED_SMEM_LB=1` and `STRATA_PIPED_SPEC_COST=1` were
reverted to `a315e6e0a` source after the online gate killed the engine on the first request, and
`STRATA_PROMPT_ATTN_Q4_IMMA=1` failed FP64 parity on `p.v` and was reverted before commit. See
[[2026-11-28-phase3-levers-rejected]] and [[2026-11-28-phase3-q4-imma-rejected]].

## Live end state

`strata-serve.service` active, `MainPID=167595`, `NRestarts=0`, `/health` `loaded: true`,
`bin=4573e5df...`, `cfg=d97f39c2...`, 0 Xids, no `nvtop`, all GPU renders `1 MiB` at every
zero-NVML gate. Raw artifacts: `/srv/lab/scratch/2026-11-27-phase3/final_qual/`.
