# BADMIT Stage 2: sticky cohort + super-pack — rejected

Date: 2026-11-27 (lab GPU host). Branch: `lever/badmit-stage2` off `41b26de66`.
Target: `burst65 c=8 warm_p50 < 400 ms`, opt-in, default path byte-identical to `main`.

## Falsifier (written before the run)

The lever ships only if the candidate's `burst65` warm p50 is below 400 ms **and** repeats below
400 ms on a second arm with the same binary/config, with 8/8 needles, 0 leaks and no new Xid.
Anything inside run-to-run noise is not a result.

## Binaries and configs

- A (baseline): installed engine sha256 `4573e5df68895b99d9d78fea37ff4bf30e3f276eb03f721b5317c09cb744ce16`
- B (candidate): rebuilt engine sha256 `2996886e389d2899f79dab095ba85c58e0d3950e62be8e3448e63666bed5b32e`
  (built Release, `--target strata batch_admit_test`, `CMAKE_CUDA_ARCHITECTURES=86;89`, gcc-13, nvcc 12.4.131)
- A config `d97f39c223b0a48f88dac349d4bc9381650844317a12c21383d9703490494cc8` (already includes
  `STRATA_BATCH_ADMIT=1`, `STRATA_BATCH_ADMIT_MAX=8`, `STRATA_BATCH_GRAPH_CAP=64`)
- B config `0849503c9008dbc5215d077dfe350e0c28b2155a994b4d70191757ada5e92762` = A +
  `STRATA_BATCH_ADMIT_STICKY_COHORT=1`, `STRATA_BATCH_ADMIT_SUPER_PACK=1`
- Isolation configs: `sp` (super only), `st` (sticky only), `m4sp` (max 4 + super), `w0` (window 0)

## Probes

`agx_agentic_burst_bench.py --context-tokens 65536 --slots 8 --turns 3 --max-tokens 64
--deltas 64,128,256` (warm p50/p95 over 24 turns), plus `burst196` (196608 tokens, 4 turns) on
the first matrix. Every arm boots through the PID-only zero-NVML teardown gate
(`pgrep -x strata-host` gone, 7 s zero-NVML settle, ONE `nvidia-smi`, 5 s settle), exports
`AGX_GPU_POLL_S=20.0`, `AGX_ENABLE_DMON=0`, and verifies the running `/proc/<pid>/exe` sha and
each flag in `/proc/<pid>/environ`. All gates: 0 new Xids, NRestarts 0, 8/8 needles, 0 leaks,
VRAM free >= 1428 MiB.

## Measured `burst65` warm p50 (ms)

| Arm | Binary | Flags | p50 | p95 | t/min | needles |
|---|---|---|---|---|---|---|
| a1 | A | baseline | 403.87 | 847.50 | 364.68 | 8/8 |
| b1 | B | sticky + super | 563.26 | 876.01 | 357.75 | 8/8 |
| a2 | A | baseline | 554.91 | 838.19 | 364.02 | 8/8 |
| b2 | B | sticky + super | 467.76 | 816.02 | 395.44 | 8/8 |
| m4sp | B | max4 + super | 588.11 | 959.07 | 360.59 | 8/8 |
| sp | B | super only | 428.57 | 842.27 | 398.29 | 8/8 |
| st | B | sticky only | 532.93 | 844.68 | 370.52 | 8/8 |
| w0 | B | window 0 | 618.18 | 827.41 | 366.40 | 8/8 |
| sp2 | B | super only | 794.39 | 859.63 | 360.78 | 8/8 |
| a3 | A | baseline | 621.71 | 842.25 | 375.16 | 8/8 |
| sp3 | B | super only | 484.81 | 831.04 | 362.48 | 8/8 |
| a4 | A | baseline | 517.77 | 843.18 | 370.61 | 8/8 |

`burst196` warm p50: a1 853.78, b1 638.02, a2 586.28, b2 740.59 ms.

## Verdict

- Sticky cohort is a regression on the target metric: its isolated arm is 532.93 ms versus
  428.57 ms for super-pack alone, and the combined arms (563/468 ms) do not beat the baseline
  repeats (555/622/518 ms). Per-turn cells show it turns some 200–380 ms turns into 480–717 ms
  turns by forcing them into larger cohorts. Its code is deleted, not left off in the tree.
- Super-pack has median 484.81 ms versus 554.91 ms for baseline, but mean 569 ms versus 525 ms
  because the *same* binary/config produced 794.39 ms on a repeat. The spread (428–794 ms) is
  wider than any lever effect, so this is noise, not a win. Its code is deleted as well.
- No arm reached the falsifier. The closest was the untouched baseline at 403.87 ms, which is
  itself above 400 ms. Target NOT met.
- Levers 3 and 4 were inspected, not implemented: in the BADMIT warm path the engine reads only
  `delta-1` tokens (the `setup=` line in the engine log's setup line is 20–41 ms when `pfx>0`), so replacing the
  O(pfx) suffix append/CRC cannot move a ~500 ms p50; `pf_hcdown_cuda()` is already enabled in
  the single-prompt path and adding it to `lm_run` changes numerics and needs a Suite A gate
  before it can be claimed. Ponytail rung 1: no speculative work.

## Evidence

Raw JSON/JSONL/logs, driver logs, configs and binary snapshots are kept in the lab scratch area
(out of repo); exact paths are in the coordinator handoff.
Offline gates during the experiment: `batch_admit_test` PASS; `check_cuda_spills.py --build-dir
build-cand` 0 not allowed on sm_86/sm_89; `tools.test_settings_register` 8/8; `capability_table.py
--check` ok; `tools/check --only docs` 6/6; `PUBLIC_BOUNDARY_HITS=0`; promoted `41b26de66` CI 4/4
success. Final live state: baseline binary and `d97f39c2` config restored, service active,
`/health` loaded, NRestarts 0, 0 new Xids.
