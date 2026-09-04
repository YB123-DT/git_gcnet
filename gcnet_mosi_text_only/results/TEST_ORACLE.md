# CMU-MOSI Text-only test-oracle diagnostic

## Purpose and validity boundary

This diagnostic retrains the unchanged strict Text-only BiGRU and evaluates the test split after every epoch. It asks how much score is lost because the small validation split selects a different epoch from the test optimum.

The oracle numbers below are **not valid benchmark results**. They inspect the test split 100 times and select the maximum. The valid result remains the validation-selected 84.51% mean in `RESULT.md`.

## Results

| Seed | Validation-selected epoch | Valid test W-F1 | Test-oracle epoch | Oracle test W-F1 | Oracle inflation |
|---:|---:|---:|---:|---:|---:|
| 66 | 59 | 82.81 | 8 | 86.42 | +3.61 |
| 67 | 23 | 84.62 | 26 | 86.97 | +2.35 |
| 68 | 12 | 85.24 | 30 | 86.87 | +1.63 |
| 69 | 36 | 84.90 | 58 | 86.36 | +1.46 |
| 70 | 27 | 84.99 | 12 | 86.45 | +1.46 |
| **Mean ± SD** | — | **84.51 ± 0.98** | — | **86.61 ± 0.28** | **+2.10 ± 0.92** |

The rerun exactly reproduces every seed's original validation-selected epoch and final test metrics. Therefore the difference is caused solely by selecting the maximum over the 100 stored test evaluations, not by a changed model, seed, or training trajectory.

## Interpretation

The MOSI validation split is a noisy proxy for test W-F1 at the epoch level. Its selected epochs differ substantially from the test-optimal epochs, particularly for seed 66. Repeatedly inspecting test reduces apparent seed variance and raises the mean by 2.10 points, which demonstrates the magnitude of test-selection bias available in this setup.

This explains why some reported or exploratory MOSI results can appear 1--3 points higher without a genuinely stronger model. It does not authorize reporting 86.61 as Text-only performance. A legitimate improvement must replace test selection with a validation-only strategy such as a predeclared smoothed validation criterion, checkpoint averaging, or stronger validation construction, and must be evaluated once on test.

Machine-readable epoch curves and metrics are stored under `results/test_oracle/`; checkpoints and prediction NPZ files are not committed.

