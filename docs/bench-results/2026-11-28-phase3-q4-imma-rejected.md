---
title: Phase 3 Q4 prompt-attention IMMA - clean rejection
truth_class: evidence
project: local-agent-lab
updated: 2026-11-28
status: "REJECTED - reverted to bae0172bf; no dormant code left in tree"
---

# Phase 3 — `STRATA_PROMPT_ATTN_Q4_IMMA=1` clean rejection

The third Phase 3 lever was attempted and rejected on the declared quality gate. Nothing from it
is committed: `lever/phase3-kernels` stays at `bae0172bf642207810cba85f53b7b8920c7b4c66`.

## What was tried

An exact-`1` opt-in Q4_0 prompt-attention kernel derived from the v3 `imma16832` path: INT8 MMA
accumulation with two independent 32-dim scale groups. Three concrete defects were found and fixed
before the gate: independent 16-byte swizzles (codes overflowed into neighbouring cells), signed
scale handling (`fabs` range), and v3's scale-at-MMA-output-column rule.

## Gate result (falsifier: parity with FP64, and no worse than baseline)

- `q.k` and logits matched FP64 to 6 decimals for heads 0–7 after the fixes.
- `p.v` still failed on the **second 32-dim block**:
  - `qsa_prompt_attn_parity` 32768/2048: IMMA vs FP64 `14.1`, vs baseline `15.5` (baseline error `8.11e-06`).
  - `qsa_prompt_attn_parity` 1500/1500: IMMA vs FP64 `11.5`, vs baseline `13.9`.
  - `FAILURES=2`.
- Synthetic timing was `1.17–1.20x`, i.e. a real speed direction — but a fast kernel that fails the
  quality gate is reverted, not tuned. Under `strata-lever-measurement` this is a hard veto.

## Disposition

Source, header and parity harness were reverted byte-for-byte to `bae0172bf`; tree clean. Offline
gates on the reverted tree: `check_cuda_spills.py` 0 not-allowed; settings register 8/8
(`python3 -m unittest tools.test_settings_register`); capability table ok; docs 6/6; links resolve;
memcheck 0 errors. No new environment variable survives in `settings-register.json` or
`docs/SETTINGS.md`.

## Unrelated host event during the same window

The first Phase 3 online A/B teardown (`systemctl stop`, PID-only wait) hung on a `strata-host`
thread in `uvm_spin_loop`; new Xid 175 (`nvtop`, GPU memory subsystem timeout) and Xid 154 (PF FLR)
followed, and the host stopped answering. This was recorded as a hardware/driver teardown fault,
**not** as a code regression from the Q4 attempt — the Q4 kernel was never loaded into the serving
binary or exercised by that run. `ai-server` was subsequently power-cycled cleanly (0 Xids, 4x RTX
4090 healthy, `nvtop` gone) before any further qualification.

## Raw artifacts

`/srv/lab/scratch/2026-11-27-phase3/` (`build_q4clean*.log`, `build_vcode*.log`, parity logs).
Board record: swarm 57e8a06a seq 36.
