# Frozen Persistent7 diagnostics

INTERNAL DIAGNOSTIC ONLY. Original Flat checkpoints (not Nested), frozen inference with no training or fine-tuning. BEST checkpoint selection retains the original test-oracle limitation.

A = Audio; T = Text; V = Video. Singles use missing-rate 0.7 BEST; pairs use 0.3 BEST; full ATV uses 0.0 BEST. Availability is fixed over each conversation and its history.

Complete seeds 66, 67, 68 are required for every dataset. For IEMOCAP4 and IEMOCAP6, first average the five folds equally within each seed; MELD has one fold. Then report the three-seed mean ± sample SD. No utterance pooling. macro6 averages the six non-full patterns within each seed; full7 includes all seven. Counts remain in the original run records.

| Pattern | MELD ACC | MELD UA | MELD WF1 | IEMOCAPFour ACC | IEMOCAPFour UA | IEMOCAPFour WF1 | IEMOCAPSix ACC | IEMOCAPSix UA | IEMOCAPSix WF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 48.480 ± 0.686 | 19.699 ± 0.978 | 39.706 ± 0.449 | 54.990 ± 1.285 | 56.938 ± 1.992 | 52.660 ± 2.493 | 44.082 ± 0.601 | 38.937 ± 0.471 | 41.624 ± 0.330 |
| T | 66.143 ± 0.246 | 47.813 ± 1.593 | 65.154 ± 0.311 | 67.443 ± 1.245 | 67.852 ± 1.543 | 67.382 ± 1.362 | 50.897 ± 0.700 | 49.553 ± 1.055 | 50.689 ± 0.368 |
| V | 47.842 ± 0.420 | 14.565 ± 0.358 | 31.994 ± 0.876 | 52.299 ± 0.386 | 49.186 ± 0.378 | 51.397 ± 0.406 | 33.373 ± 0.714 | 31.720 ± 0.584 | 30.419 ± 0.740 |
| AT | 66.526 ± 0.775 | 45.203 ± 1.965 | 65.045 ± 0.526 | 76.438 ± 1.294 | 77.670 ± 1.024 | 76.357 ± 1.348 | 60.656 ± 0.716 | 59.000 ± 1.126 | 60.282 ± 0.716 |
| AV | 49.425 ± 0.703 | 17.586 ± 2.186 | 37.323 ± 3.478 | 65.768 ± 1.985 | 65.201 ± 2.689 | 65.704 ± 1.998 | 47.764 ± 2.268 | 45.272 ± 2.912 | 46.414 ± 2.735 |
| TV | 66.654 ± 0.160 | 45.133 ± 2.472 | 65.053 ± 0.442 | 75.061 ± 0.255 | 74.130 ± 0.438 | 74.778 ± 0.232 | 56.203 ± 1.415 | 54.636 ± 0.609 | 55.952 ± 1.202 |
| ATV | 66.986 ± 0.463 | 43.303 ± 1.596 | 65.021 ± 0.726 | 82.078 ± 0.340 | 82.292 ± 0.460 | 81.989 ± 0.301 | 64.612 ± 0.264 | 62.962 ± 0.563 | 64.359 ± 0.022 |
| macro6 | 57.512 ± 0.013 | 31.666 ± 1.007 | 50.712 ± 0.515 | 65.333 ± 0.261 | 65.163 ± 0.342 | 64.713 ± 0.475 | 48.829 ± 0.501 | 46.520 ± 0.550 | 47.563 ± 0.389 |
| full7 | 58.865 ± 0.072 | 33.329 ± 0.873 | 52.756 ± 0.434 | 67.725 ± 0.272 | 67.610 ± 0.287 | 67.181 ± 0.449 | 51.084 ± 0.467 | 48.869 ± 0.507 | 49.963 ± 0.336 |

## Original random-missing reference (not Persistent)

| Rate | MELD ACC | MELD UA | MELD WF1 | IEMOCAPFour ACC | IEMOCAPFour UA | IEMOCAPFour WF1 | IEMOCAPSix ACC | IEMOCAPSix UA | IEMOCAPSix WF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.986 | 43.303 | 65.021 | 82.078 | 82.292 | 81.989 | 64.612 | 62.962 | 64.359 |
| 0.1 | 65.645 | 44.484 | 64.053 | 81.750 | 81.953 | 81.675 | 63.796 | 61.957 | 63.522 |
| 0.2 | 64.023 | 40.978 | 61.869 | 80.788 | 80.961 | 80.662 | 63.297 | 61.247 | 63.095 |
| 0.3 | 62.439 | 39.475 | 59.981 | 80.173 | 80.311 | 80.066 | 62.542 | 60.848 | 62.271 |
| 0.4 | 60.881 | 36.743 | 57.519 | 79.369 | 79.704 | 79.294 | 61.635 | 59.890 | 61.119 |
| 0.5 | 59.246 | 33.923 | 55.364 | 78.322 | 78.894 | 78.210 | 60.608 | 59.398 | 60.176 |
| 0.6 | 57.152 | 32.430 | 52.937 | 76.841 | 77.040 | 76.619 | 59.443 | 57.988 | 59.058 |
| 0.7 | 56.079 | 30.484 | 51.324 | 75.488 | 75.900 | 75.345 | 57.780 | 56.520 | 57.297 |
| Mean8 | 61.557 | 37.728 | 58.509 | 79.351 | 79.632 | 79.233 | 61.714 | 60.101 | 61.362 |

MELD remains lr1e-5/wd1e-5/50epochs; IEMOCAP remains original lr1e-3/wd1e-5/100epochs. No protocol unification by retraining.
MELD A/T/V reuse the just-completed baseline evaluation; full ATV reuses the verified all-visible selected replay.

## Execution and verification

- Complete: 33 seed/fold units and 231 fixed-pattern records. All three workers
  exited successfully. Every pattern's saved logits, labels and fixed availability
  were checked offline; ACC/UA/W-F1/CE were recomputed and matched the report.
- 99 exact selected-mask replays, including reused checks, cover three source
  rates per seed/fold. Newly replayed logits have maximum absolute error **0**
  against their retained original predictions. No new random masks were sampled.
- Original numerical sources remain MELD `b4a3138` and IEMOCAP `785a012` in
  `cmerc-repro`. No model weights, training protocol or historical result files
  were changed. Checkpoint/source hashes were verified before and after evaluation.
- MELD evaluation source: `5fe78a3`; IEMOCAP evaluation source: `1e299d0`.
  The first IEMOCAP attempt stopped before inference at a JSON-list versus
  dataclass-tuple configuration guard. The correction normalizes serialization
  for comparison only; the historical model/config stay unchanged. Complete
  MELD results were reused only after confirming unchanged evaluator and baseline
  hashes. Failed attempt logs remain in `runs_v1`; no failed/partial scores enter
  the means.
- Actual frozen worker times: MELD 32.23 s, IEMOCAP4 133.24 s, IEMOCAP6 138.65 s;
  peak allocated GPU memory 205.27, 575.17, 573.46 MiB respectively. Physical GPU3
  only; no training or additional GPU experiment was started.
- Runtime outputs, full class recalls/supports and predictions:
  `/home/qinjinghui/remote_experiments/osram_persistent_patterns_20261018/runs_v2`.
  Reused MELD artifacts retain their original `runs_v1`/unimodal paths.
- `SUMMARY.json` records configurations, source checkpoints, selected epochs,
  mask hashes and aggregations. `VERIFICATION.json` records artifact hashes and
  completed checks; CSVs retain all individual seed/fold scores.
- Automated tests: 32 passed, including the tuple-restoration regression.

Finite continuation command used after preserving the first attempt:

```bash
cd /home/qinjinghui/remote_experiments/osram_persistent_patterns_20261018/code_1e299d0
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_persistent_patterns_20261018/run_all.py \
 --output /home/qinjinghui/remote_experiments/osram_persistent_patterns_20261018/runs_v2 \
 --commit 1e299d02891faac32cc44f289ebc440994a4190f \
 --reuse-meld /home/qinjinghui/remote_experiments/osram_persistent_patterns_20261018/runs_v1/MELD
```
