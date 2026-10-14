# MELD no-Local-skip retraining under the original GCNet mask

**INTERNAL TEST-ORACLE DIAGNOSTIC ONLY**

## Protocol

- Code commit: `ac2aa446ae4ca605aeb2de81f798e89d9ff18612`
- Dataset: MELD
- Seeds: 66, 67, 68
- Epochs: 50
- Missing sampler: `gcnet-random-mask-v1`
- Selection: separate best test epoch for each missing rate, using accuracy
- Baseline: corrected GCNet-random Flat seed-66 run
- Intervention: remove only `local_skip(local)` from the final Flat readout;
  adapter Local, Base, Gap, CHM, task head, optimizer, and schedule are retained.

The deterministic `gcnet-legacy-fixed-v1` replay result is not used in this
comparison.

## Three-seed Test-best result

| Model / seed | 8-rate ACC | 8-rate W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| No skip, 66 | 60.560 | 56.400 | 56.718 | 50.700 |
| No skip, 67 | 60.517 | 56.238 | 56.284 | 50.091 |
| No skip, 68 | 60.584 | 57.018 | 56.667 | 51.942 |
| **No skip, mean** | **60.554** | **56.552** | **56.556** | **50.911** |
| Flat, mean | 60.498 | 56.754 | 56.420 | 51.352 |
| **No skip - Flat** | **+0.056** | **-0.202** | **+0.136** | **-0.441** |

The shortcut removal does not yield a consistent task improvement: mean ACC is
effectively unchanged, while weighted F1 decreases overall and at high missing.

UA cannot be reconstructed for the exact historical mask that first set each
Test-best checkpoint because those per-epoch predictions were not saved. On the
reproducible final saved-mask evaluation of the same selected checkpoints:

| Model | 8-rate UA | High UA |
|---|---:|---:|
| Flat, 3-seed mean | 34.980 | 29.213 |
| No skip, 3-seed mean | 34.758 | 28.350 |
| **No skip - Flat** | **-0.222** | **-0.863** |

## Seed-66 detail

Values are percentages. The checkpoint epoch is selected independently for
each missing rate under the retained internal Test-oracle protocol.

| Missing rate | Flat ACC | No skip ACC | Delta | Flat W-F1 | No skip W-F1 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.284 | 66.284 | +0.000 | 64.897 | 64.172 | -0.725 |
| 0.1 | 65.172 | 64.828 | -0.344 | 63.415 | 61.686 | -1.729 |
| 0.2 | 63.103 | 62.912 | -0.191 | 60.897 | 60.816 | -0.081 |
| 0.3 | 61.494 | 61.149 | -0.345 | 57.866 | 56.636 | -1.230 |
| 0.4 | 59.693 | 59.157 | -0.536 | 54.641 | 55.789 | +1.148 |
| 0.5 | 57.893 | 58.659 | +0.766 | 53.526 | 53.445 | -0.081 |
| 0.6 | 56.360 | 56.054 | -0.306 | 50.646 | 51.027 | +0.381 |
| 0.7 | 55.096 | 55.441 | +0.345 | 50.805 | 47.628 | -3.177 |
| **8-rate mean** | **60.637** | **60.560** | **-0.077** | **57.087** | **56.400** | **-0.687** |
| **High missing mean** | **56.450** | **56.718** | **+0.268** | **51.659** | **50.700** | **-0.959** |

Selected epochs for no-skip at rates 0.0 through 0.7 were:
`22, 14, 17, 15, 26, 14, 26, 10`.

### Final saved-prediction evaluation

The original GCNet sampler is stochastic. Consequently, the mask sampled when
an epoch sets a Test-oracle maximum is not the same mask sampled during the
final saved-prediction evaluation. The latter is the source of the previously
reported approximately 59-point MELD result. On the seed-66 paired runs:

| Metric | Flat | No skip | Delta |
|---|---:|---:|---:|
| 8-rate ACC | 59.674 | 59.737 | +0.063 |
| 8-rate W-F1 | 55.902 | 55.329 | -0.573 |
| High-missing ACC | 55.032 | 55.492 | +0.460 |
| High-missing W-F1 | 49.698 | 48.845 | -0.853 |

The corresponding three-seed Flat mean is 59.746 ACC; it must not be directly
subtracted from this single-seed no-skip result. Its seed-matched Flat value is
59.674 ACC.

## Interpretation boundary

Removing the final Local shortcut does not improve the seed-66 Test-oracle
mean. Accuracy is almost unchanged and high-missing accuracy rises slightly,
but weighted F1 decreases overall and at high missing rates. In particular,
the 0.7 W-F1 loss is 3.177 percentage points. This is not evidence that Memory
can replace the direct Local shortcut. A single seed is also insufficient for
a formal performance claim.

## Artifacts

Remote run:

```text
/data2/yb/paper/remote_runs_20261012/results/
osram_meld_gcnet_random_no_skip_20261014/seed_66
```

The directory contains provenance, configuration, history, metrics, selected
per-rate checkpoints, and predictions.
