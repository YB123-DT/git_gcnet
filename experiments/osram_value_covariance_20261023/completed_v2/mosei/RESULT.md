# Frozen Value-noise diagnostic

INTERNAL DIAGNOSTIC ONLY. No training.

| Dataset | Rate | sigma | N | Clean ACC | Noisy ACC | ΔMSE | Mean absolute prediction shift | Flip % | Corrections / harms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMUMOSEI | 0.3 | 0.001 | 4659 | 84.7001 | 84.7104 | 1.02579e-08 | 4.13772e-06 | 0.0103192 | 0.375 / 0 |
| CMUMOSEI | 0.3 | 0.01 | 4659 | 84.7001 | 84.7155 | 1.26411e-07 | 4.13682e-05 | 0.0154788 | 0.5625 / 0 |
| CMUMOSEI | 0.7 | 0.001 | 4659 | 78.1233 | 78.1198 | 9.51743e-08 | 5.81634e-06 | 0.00343974 | 0 / 0.125 |
| CMUMOSEI | 0.7 | 0.01 | 4659 | 78.1233 | 78.1112 | 1.02222e-06 | 5.81656e-05 | 0.0120391 | 0 / 0.4375 |

Noise is applied to actual historical projected Values, not separately to Base/Gap. The same unit noise is reused at both sigmas.
Binary metrics/counts exclude y=0 and retain threshold prediction>0; MSE includes all valid labels.
The head-only intervention uses draw0; it is not a Monte Carlo estimate. Kappa is not task uncertainty.
AGGREGATE.json contains pattern, correctness, history, kappa-quartile and head strata. STATUS.json contains parity and covariance errors.
Full G matrices and per-draw predictions are retained in remote batch NPZ/ROWS artifacts, not committed to Git.
