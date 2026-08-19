# Modality-JEPA Session-5 all missing rates, 10 trials

IEMOCAP-Six Session 5, seeds 66--75, ten complete training trials per nonzero
missing rate. Missing=0 reuses the identical GCNet parity runs. Both methods use
the same training configuration; current results retain inherited test-as-validation
selection and are diagnostic rather than leakage-free final scores.

| Missing | GCNet W-F1 | JEPA W-F1 | Paired delta | JEPA wins | Paired t-test p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 63.37 +/- 2.05 | 63.37 +/- 2.05 | +0.00 | parity | 1.0000 |
| 0.1 | 64.49 +/- 1.11 | 63.94 +/- 1.62 | -0.55 | 4/10 | 0.3713 |
| 0.2 | 63.23 +/- 1.54 | 62.37 +/- 1.67 | -0.86 | 4/10 | 0.2503 |
| 0.3 | 62.43 +/- 1.82 | 62.84 +/- 1.30 | +0.40 | 6/10 | 0.5749 |
| 0.4 | 61.78 +/- 1.95 | 61.73 +/- 1.89 | -0.05 | 6/10 | 0.9591 |
| 0.5 | 60.68 +/- 1.46 | 61.63 +/- 1.76 | +0.95 | 6/10 | 0.1524 |
| 0.6 | 59.96 +/- 2.20 | 59.80 +/- 2.60 | -0.16 | 7/10 | 0.8882 |
| 0.7 | 60.63 +/- 2.44 | 60.50 +/- 2.20 | -0.13 | 6/10 | 0.8890 |

The average over rates is 62.02 for JEPA and 62.07 for GCNet. No individual
rate has a statistically significant paired improvement at alpha=0.05. The
largest positive mean delta is +0.95 at missing=0.5, but its paired t-test
p-value is 0.1524. These experiments therefore do not establish an ERC benefit
from the current modality prediction auxiliary objective.

Machine-readable per-seed metrics and diagnostic latents are stored in
`summary_all_rates.json`.
