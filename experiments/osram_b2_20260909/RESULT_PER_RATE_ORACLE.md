# Per-rate Test-oracle re-extraction

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

Each seed and each missing rate independently selects the highest test weighted-F1 among epochs 1–100. Ties select the earliest epoch. Other metrics come from that same selected epoch. This is an optimistic multi-epoch diagnostic, not one deployable checkpoint's eight-rate performance.

No retraining or model evaluation was performed. Existing best.pt files are unchanged; these are logged metrics, not a claim that all selected epoch checkpoints were retained. P0 and B2 evaluation mask hashes match for all five seeds. B2 has extra pretraining/fine-tuning budget; this is not an equal-budget ablation.

## Five-seed mean W-F1 (%)

| Rate | P0 | B2 | Delta (pp) | Positive seeds |
|---|---:|---:|---:|---:|
| 0.0 | 87.4168 | 87.1305 | -0.2862 | 0/5 |
| 0.1 | 85.3979 | 84.9266 | -0.4713 | 1/5 |
| 0.2 | 81.9353 | 82.2436 | +0.3083 | 3/5 |
| 0.3 | 80.9071 | 80.6651 | -0.2420 | 2/5 |
| 0.4 | 78.0876 | 78.3125 | +0.2249 | 3/5 |
| 0.5 | 76.6299 | 76.0082 | -0.6218 | 1/5 |
| 0.6 | 75.0903 | 75.1265 | +0.0362 | 4/5 |
| 0.7 | 74.5528 | 73.6539 | -0.8989 | 2/5 |
| Eight-rate mean | 80.0022 | 79.7584 | -0.2439 | — |

## Per-seed eight-rate mean and selected epochs

Epoch lists follow rates 0.0 through 0.7.

| Seed | P0 mean | B2 mean | Delta (pp) | P0 epochs | B2 epochs |
|---|---:|---:|---:|---|---|
| 66 | 80.2476 | 80.6623 | +0.4148 | [30, 30, 51, 30, 47, 48, 30, 29] | [29, 17, 16, 15, 47, 29, 17, 17] |
| 67 | 80.2816 | 79.6648 | -0.6168 | [41, 41, 41, 54, 41, 54, 93, 59] | [18, 18, 18, 35, 19, 32, 95, 18] |
| 68 | 79.9461 | 79.0310 | -0.9151 | [39, 28, 47, 47, 47, 26, 31, 48] | [6, 10, 8, 16, 10, 11, 5, 42] |
| 69 | 80.3263 | 80.0416 | -0.2847 | [46, 92, 46, 36, 37, 58, 91, 56] | [7, 7, 7, 7, 12, 27, 27, 27] |
| 70 | 79.2094 | 79.3920 | +0.1826 | [46, 53, 51, 56, 78, 46, 65, 53] | [10, 9, 10, 44, 26, 26, 26, 44] |

P0 high-missing (0.5/0.6/0.7) mean: 75.4244%.

B2 high-missing (0.5/0.6/0.7) mean: 74.9295%.

Full per-seed/rate metrics (fractions for F1/accuracy): `per_rate_oracle.csv`. Original histories and hashes are retained for reproducibility. The previous `RESULT.md` remains the explicitly labeled eight-rate-mean selection report; do not mix the two protocols.
