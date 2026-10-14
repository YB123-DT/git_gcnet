# MELD no-Local-skip retraining under the original GCNet mask

**INTERNAL TEST-ORACLE DIAGNOSTIC ONLY**

## Protocol

- Code commit: `ac2aa446ae4ca605aeb2de81f798e89d9ff18612`
- Dataset: MELD
- Seed: 66
- Epochs: 50
- Missing sampler: `gcnet-random-mask-v1`
- Selection: separate best test epoch for each missing rate, using accuracy
- Baseline: corrected GCNet-random Flat seed-66 run
- Intervention: remove only `local_skip(local)` from the final Flat readout;
  adapter Local, Base, Gap, CHM, task head, optimizer, and schedule are retained.

The deterministic `gcnet-legacy-fixed-v1` replay result is not used in this
comparison.

## Results

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
