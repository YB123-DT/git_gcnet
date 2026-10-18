# TASR verification before finite screening

INTERNAL DIAGNOSTIC ONLY. Smoke is implementation verification, not evidence of
performance improvement. Core implementation commit: `1a56a21`.

Tests-first: integration initially failed on six missing opt-in/API tests;
source primitives initially failed on seven absent primitives; protocol tests
initially failed14. All were implemented and rerun. A runtime Gap-ablation
leak was reproduced by two new failing tests, then fixed and verified.

Focused current run: **81 passed,1 deselected** (Torch1.13-compatible environment).
The deselected historical `test_fractional_eval_matches_frozen_intervention`
uses `register_forward_pre_hook(...,with_kwargs=True)`, unsupported by Torch1.13.
Running that test against pre-TASR `ea27c08` also failed with the SAME TypeError;
it is not silently counted as passing or repaired outside this task.

## Real protected-checkpoint smoke

Server: biggpu. Environment: `/data2/yb/conda_envs/cmerc-repro/bin/python`,
Python3.10.13 / Torch1.13.1+cu117 / NumPy1.26.4 / sklearn1.3.2.
GPU0(MELD) and GPU5(IEMOCAP6) bound by real UUID; broken physicalGPU4 excluded.
Full provenance, original weight/selected-mask hashes, configurations and splits:
`SMOKE_MELD.json`, `SMOKE_IEMOCAPSix.json`.

| Dataset | Memory max abs | Memory max relative | Gap max abs | Gap max relative | Peak MiB | Wall sec |
|---|---:|---:|---:|---:|---:|---:|
| MELD | 3.5763e-7 | 2.1188e-7 | 2.5332e-7 | 2.6846e-7 | 308.50 |19.81|
| IEMOCAP6 fold5 |1.7881e-6|2.8777e-7|1.9073e-6|1.7447e-7|1020.88|62.95|

Table errors use the real selected-mask first test batch; all three paired
training-batch forward/backward updates and their errors are also in JSON.
Relative errors are Frobenius-error / reference Frobenius-norm (clamped1e-12),
maximized over visited valid timesteps/target slots, not per-element ratios.

Both datasets passed:

- Current code strictly loads protected original `.3` state dict and reproduces
  its COMPLETE saved `.3` cohort logits with maximum absolute error **0**.
  No historical-class bridge, partial load or substitute weight was necessary.
- First-batch real-source AND neutral-source Adapter zero-init logits agree
  exactly with the original model; CPU/CUDA eval and initialization RNG unchanged.
- First utterance's source Gap reads strictly zero. Original shared Memory stays
  authoritative; source/read sum errors match float32 roundoff above.
- Exactly59,423 added/trainable parameters; every original parameter frozen,
  model remains eval; original backbone digest unchanged after updates.
- Three optimizer updates use TRAINING labels only; each task loss finite.
  First step:12 parameter tensors connected,2 nonzero-gradient tensors (final
  layer). Steps2/3:12 connected AND12 nonzero, all finite. This explicitly
  verifies upstream learning after zero-init rather than treating zero as success.
- Adapter/optimizer/RNG checkpoint restores strictly; recomputed logits exactly
  match pre-save logits. Normal and independent whole-conversation Persistent
  views use separate zero-state forward scans, without NaN/Inf.

Additional controlled FP64 algebra: Memory abs6.66e-16/rel2.82e-16;
Gap abs6.66e-16/rel3.77e-16. FP32 controlled algebra: Memory abs3.58e-7/rel1.56e-7;
Gap abs4.77e-7/rel1.73e-7. These do NOT imply semantic purity of sources.

## Finite next stage

Correctness gate is passed; retain four jobs only: MELD F1/F2 and IEMOCAP6
seed66/fold5 F1/F2,30epochs. One fixed `.3` original checkpoint each; single
Adapter checkpoint chosen using only declared validation ACC/CE. No test-driven
selection, no extra folds/seeds/search. Immutable run source will be pinned after
this verification commit, with unique outputs and actual mask/RNG records.
