# Paired task-only transfer: MELD and IEMOCAP6

INTERNAL TEST-ORACLE SCREEN. Seed66 only; IEMOCAP6 means require all five folds.
Status: complete

## Completed comparison summary

All comparisons below use **seed66 against seed66**, not the three-seed MELD
baseline average. IEMOCAP6 averages all five completed folds equally. Changes
are percentage points; only one backbone seed was tested.

| Dataset | Evaluation | Delta ACC | Delta UA | Delta W-F1 |
|---|---|---:|---:|---:|
| MELD | Random8 | -0.211 | +0.552 | +0.037 |
| MELD | High random .5/.6/.7 | -0.524 | -0.045 | -0.367 |
| MELD | Persistent six-pattern mean | +0.083 | +0.291 | +0.220 |
| IEMOCAP6 | Random8 | -0.278 | -0.171 | +0.087 |
| IEMOCAP6 | High random .5/.6/.7 | -0.266 | -0.673 | +0.118 |
| IEMOCAP6 | Persistent six-pattern mean | +0.805 | +0.501 | +1.038 |

This transfer does not reproduce the large MOSI persistent gain. MELD persistent
changes are small. IEMOCAP6 shows a modest six-pattern benefit, not improvement
in every condition: V-only improves, whereas AV declines in ACC/UA. Random ACC
falls slightly on both datasets; improved UA on MELD does not mean both metrics
improved. These are exploratory observations, not statistical significance or
proof of improved Memory storage/retrieval. No additional seeds or configurations
were launched based on these results.

## Verification and retained evidence

- Six of six jobs completed; MELD50epochs and IEMOCAP6 five folds100epochs each,
  totaling550 committed training epochs. First-epoch checks and complete
  last_training states are preserved along with eight BEST per job.
- Independently checked all48 selected checkpoint hashes, exact selected
  masks/predictions, and all42 fixed-pattern prediction sets. ACC/UA/W-F1/CE
  recomputation matches. All18 original selected-mask frozen replays have zero
  maximum logit error. No training or new model inference was repeated by the audit.
- All six final last_training states contain matching experiment identity,
  final epoch, nonempty optimizer state and a complete one-device CUDA RNG state.
  Fivefold and pattern aggregates were independently recomputed.
- Source driver: `654f67f18a9cfb4297080e8b9eb9a7dbb35c74bf`. Numerical sources
  remain MELD b4a3138 and IEMOCAP785a012 in cmerc-repro. The packaging-only recovery
  documented in LAUNCH.md remains recorded; there was no numerical/config change
  or restart of the running MELD model.
- Full remote outputs:
  `/home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/runs_v1`.
  `RESULTS.json` contains fold results, source identities/configurations, selected
  epochs and masks/hashes; `per_fold_random.csv` and `per_fold_persistent.csv`
  provide48/42 individual rows. `VERIFICATION.json` records the completed audit.
- No model weights, feature data, dense prediction NPZ or full optimizer states
  are committed to Git. They remain in the protected remote experiment outputs.

## Full result tables

Existing rho=.5, history_drop_prob=.2, no contrastive loss. Original no-JEPA Flat/CHM unchanged.
MELD lr1e-5/wd1e-5/50epochs; IEMOCAP6 lr1e-3/wd1e-5/100epochs; batch32; gcnet-random-mask-v1.
Task CE supervises ALL valid utterances in BOTH views; same-current/different-history anchors do not restrict task loss.
Inference uses one view; no training-time InfoNCE/projector call at inference.
View2 is only a nonempty subset of View1, not persistent-pattern training.

| Dataset | Random rate | Flat ACC | Paired ACC | Flat UA | Paired UA | Flat WF1 | Paired WF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| MELD | 0.0 | 67.510 | 67.548 | 44.330 | 44.216 | 65.691 | 65.689 |
| MELD | 0.1 | 65.709 | 65.747 | 45.581 | 45.987 | 64.245 | 64.534 |
| MELD | 0.2 | 64.100 | 63.946 | 40.928 | 42.910 | 61.880 | 62.188 |
| MELD | 0.3 | 62.261 | 62.414 | 40.207 | 40.915 | 59.996 | 60.310 |
| MELD | 0.4 | 60.920 | 60.728 | 38.847 | 40.412 | 58.209 | 58.698 |
| MELD | 0.5 | 59.770 | 59.502 | 33.188 | 33.207 | 55.672 | 55.535 |
| MELD | 0.6 | 57.433 | 56.705 | 32.255 | 32.448 | 53.141 | 52.748 |
| MELD | 0.7 | 56.590 | 56.015 | 32.026 | 31.679 | 52.285 | 51.715 |
| MELD | mean | 61.786 | 61.576 | 38.420 | 38.972 | 58.890 | 58.927 |
| MELD | high | 57.931 | 57.407 | 32.490 | 32.445 | 53.699 | 53.333 |
| IEMOCAPSix | 0.0 | 64.902 | 64.282 | 62.363 | 62.106 | 64.385 | 64.165 |
| IEMOCAPSix | 0.1 | 64.293 | 63.972 | 61.665 | 61.574 | 63.760 | 63.677 |
| IEMOCAPSix | 0.2 | 63.011 | 63.138 | 61.062 | 61.758 | 62.526 | 63.115 |
| IEMOCAPSix | 0.3 | 62.969 | 62.613 | 61.411 | 60.740 | 62.614 | 62.157 |
| IEMOCAPSix | 0.4 | 61.828 | 61.574 | 59.520 | 60.493 | 61.054 | 61.570 |
| IEMOCAPSix | 0.5 | 61.423 | 60.569 | 59.859 | 58.722 | 60.729 | 60.418 |
| IEMOCAPSix | 0.6 | 59.740 | 59.251 | 57.844 | 57.557 | 59.045 | 58.736 |
| IEMOCAPSix | 0.7 | 57.925 | 58.469 | 55.835 | 55.241 | 56.944 | 57.918 |
| IEMOCAPSix | mean | 62.011 | 61.734 | 59.945 | 59.774 | 61.382 | 61.469 |
| IEMOCAPSix | high | 59.696 | 59.430 | 57.846 | 57.173 | 58.906 | 59.024 |

Random per-rate BEST scores retain the original resampled Test-oracle selection protocol; do not call them a common-mask test.

| Dataset | Persistent observed set | Flat ACC | Paired ACC | Flat UA | Paired UA | Flat WF1 | Paired WF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| MELD | A | 48.084 | 48.199 | 20.791 | 21.137 | 40.104 | 40.577 |
| MELD | T | 66.322 | 66.015 | 48.991 | 48.719 | 65.512 | 65.152 |
| MELD | V | 48.084 | 48.123 | 14.323 | 14.334 | 31.393 | 31.403 |
| MELD | AT | 67.011 | 66.858 | 46.868 | 46.815 | 65.648 | 65.544 |
| MELD | AV | 49.119 | 50.268 | 17.715 | 18.640 | 37.972 | 39.294 |
| MELD | TV | 66.475 | 66.130 | 46.734 | 47.526 | 65.215 | 65.193 |
| MELD | ATV | 67.510 | 67.548 | 44.330 | 44.216 | 65.691 | 65.689 |
| MELD | macro_six | 57.516 | 57.599 | 32.570 | 32.862 | 50.974 | 51.194 |
| MELD | macro_seven | 58.944 | 59.020 | 34.250 | 34.484 | 53.077 | 53.265 |
| IEMOCAPSix | A | 44.489 | 45.287 | 39.222 | 38.472 | 41.901 | 42.970 |
| IEMOCAPSix | T | 51.171 | 52.601 | 48.335 | 50.076 | 50.658 | 51.734 |
| IEMOCAPSix | V | 34.195 | 36.733 | 31.047 | 33.544 | 30.978 | 34.433 |
| IEMOCAPSix | AT | 61.185 | 61.524 | 60.219 | 59.543 | 60.813 | 60.734 |
| IEMOCAPSix | AV | 48.563 | 47.770 | 46.397 | 44.503 | 47.139 | 47.076 |
| IEMOCAPSix | TV | 56.597 | 57.112 | 53.995 | 56.085 | 56.112 | 56.882 |
| IEMOCAPSix | ATV | 64.902 | 64.282 | 62.363 | 62.106 | 64.385 | 64.165 |
| IEMOCAPSix | macro_six | 49.367 | 50.171 | 46.536 | 47.037 | 47.934 | 48.972 |
| IEMOCAPSix | macro_seven | 51.586 | 52.187 | 48.797 | 49.190 | 50.284 | 51.142 |

Persistent support is fixed across each whole conversation. Singles use .7 BEST, pairs .3, full .0.
These are not seven independently trained models. No performance claim before all relevant folds complete.
Checkpoints, exact selected masks/predictions, full optimizer/RNG state and persistent predictions are retained remotely.
The inherited inert training projector is initialized in fork_rng, receives no task-only gradient, and is excluded from backbone initialization comparisons.
Existing tests passed in both original numerical checkouts; first committed training epochs validate real CE/backward/optimizer/saving and memory admission.
