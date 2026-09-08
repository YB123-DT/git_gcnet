# Forward-only OSRAM: completed MOSI diagnostic

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.
Selection: independent maximum test weighted-F1 for each rate and seed.
All five seeds (66–70) have 100 history entries and final metrics.json.
Reference: inherited osram_heads8_out700_20260906/per_seed_rate.csv.

| Rate | Bidirectional W-F1 % | Forward-only W-F1 % | Delta pp |
|---|---:|---:|---:|
| 0.0 | 87.639 | 87.212 | -0.428 |
| 0.1 | 85.365 | 84.909 | -0.456 |
| 0.2 | 82.693 | 81.861 | -0.832 |
| 0.3 | 81.439 | 80.482 | -0.957 |
| 0.4 | 79.350 | 78.062 | -1.288 |
| 0.5 | 77.300 | 76.438 | -0.863 |
| 0.6 | 76.804 | 75.054 | -1.750 |
| 0.7 | 75.654 | 73.744 | -1.910 |

Forward-only seed eight-rate means (%): 79.7457, 80.1493, 79.6187,
79.9413, 79.1454. Overall 79.7201%, versus bidirectional 80.7806%.

Interpretation: removing future memory reduces scores under this configuration,
especially high missing rates. This does not prove causal models cannot work;
zeroing reverse slots also removes effective contextual capacity. Preserve the
bidirectional reference, and do not add fixes without a new experimental decision.

Raw histories/configs/metrics are archived in raw/seed_*/. No baseline retraining.
