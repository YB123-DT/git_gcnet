# MELD fixed-rate specialists — completed

INTERNAL SINGLE-RATE TEST-ORACLE SCREEN. Seed66 only.

All eight independent models completed 50/50 epochs; no failed jobs. Each model
trained and tested only at its own missing rate. This is not persistent missing,
not a fixed-mask experiment, and not one model evaluated across eight rates.

## ACC-selected results (%)

UA, W-F1 and CE come from the exact same saved predictions/masks as the selected
Test ACC, not a freshly sampled final retest. UA is seven-class macro recall.

| Train/test rate | Selected epoch | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|---:|
| 0.0 | 5 | 66.782 | 44.573 | 64.882 | 0.997368 |
| 0.1 | 10 | 64.828 | 45.598 | 63.261 | 1.090936 |
| 0.2 | 9 | 62.912 | 39.429 | 60.185 | 1.163794 |
| 0.3 | 13 | 61.111 | 38.452 | 58.977 | 1.183841 |
| 0.4 | 14 | 59.962 | 35.706 | 56.630 | 1.249007 |
| 0.5 | 8 | 58.582 | 32.850 | 54.108 | 1.264322 |
| 0.6 | 17 | 57.011 | 31.106 | 52.262 | 1.252841 |
| 0.7 | 30 | 55.900 | 28.841 | 50.387 | 1.379589 |
| Eight-rate mean | — | **60.886** | **37.069** | **57.586** | 1.197712 |
| High (.5/.6/.7) mean | — | **57.165** | **30.932** | **52.252** | 1.298917 |

Each rate includes all 2610 test utterances. The .7 sampler retains one modality
per valid utterance, so its actual missing fraction is 2/3 under original GCNet.
Full precision, selected checkpoint/artifact/mask hashes, per-class recall and
support are in FINAL_SUMMARY.json.

## Historical cyclic comparison, not exact-mask paired

The recent W0 constant-LR cyclic control has the same seed and main training
hyperparameters (see osram_meld_write_tuning_20261017/RESULT.md), but was trained
from a different source snapshot and does not share these selected test masks.

| Training | Eight-rate ACC | UA | W-F1 |
|---|---:|---:|---:|
| W0: one cyclic model, seed66 | 60.838 | 35.633 | 56.789 |
| Eight fixed-rate specialists, seed66 | 60.886 | 37.069 | 57.586 |
| Difference, pp | +0.048 | +1.437 | +0.798 |

The accepted historical Flat seed66 is ACC60.675 / UA35.436 / W-F1 56.807;
the accepted three-seed mean remains ACC60.693 / UA36.255 / W-F1 57.045.
Neither is replaced by this single-seed specialist experiment.

Interpretation: dedicated-rate training did not produce a material aggregate
ACC gain versus W0. UA and W-F1 are higher descriptively, but this is one seed,
test-oracle selection, unmatched masks and **eight times the total training
updates** (8x50 model-epochs and 13200 steps versus 50 epochs and 1650 steps).
It is not evidence of an equal-compute improvement or a new deployable robust
single model. No significance or mechanism claim is made.

## Configuration, artifacts and checks

- Numerical source: `2bdaf2cc5b0acc6eaf787274d59565c08aca0b02`; same 135-file
  SOURCE_MANIFEST for all workers. Model initialization hashes match across eight
  runs; parameter count is unchanged at 13,290,795 each.
- MELD ECERC canonical train/dev/test; dataset SHA256
  `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.
- Original Flat1600 plus Local Skip, causal OSRAM, no-JEPA/emotion-only,
  write_step=.6, 8 heads, K/V64; no Memory, Query, write or loss modification.
- Original `gcnet-random-mask-v1`; batch32, Adam LR=.001 constant, weight
  decay1e-5, dropout=.5. Only training-rate mode is changed to fixed.
- biggpu GPU0/5, four jobs per GPU, UUID-bound; GPU4 excluded. Original
  `/data2/yb/conda_envs/cmerc-repro/bin/python`, Torch1.13.1+cu117/NumPy1.26.4.
- Every epoch committed exactly33 same-rate optimizer steps, 1038 training
  conversations, only its own test rate and JEPA loss0. Each job retains its
  ACC-best checkpoint, exact selected predictions/mask and full last_training.pt.
- CPU regression suite previously passed28 tests (launch record); final artifact
  verification recomputed metrics from stored logits and verified checkpoint,
  prediction and mask hashes for all8. No new model inference or training was
  performed to produce this report.
- Checkpoints/large predictions remain remote and are not committed. Output:
  `/home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/runs_v1`.
- Reproduction command and source deployment are in README.md and LAUNCH.md.
  The authorized finite queue is complete; no extra seeds or search launched.
