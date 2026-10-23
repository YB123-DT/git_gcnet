# Frozen Value-noise diagnostic

INTERNAL DIAGNOSTIC ONLY. No training.

| Dataset | Rate | sigma | N | Clean ACC | Noisy ACC | ΔMSE | Mean absolute prediction shift | Flip % | Corrections / harms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMUMOSI | 0.3 | 0.001 | 686 | 80.7927 | 80.7927 | 1.20099e-06 | 4.34798e-05 | 0 | 0 / 0 |
| CMUMOSI | 0.3 | 0.01 | 686 | 80.7927 | 80.7927 | 1.21869e-05 | 0.000434814 | 0 | 0 / 0 |
| CMUMOSI | 0.7 | 0.001 | 686 | 67.2256 | 67.2256 | -3.05616e-06 | 6.9692e-05 | 0 | 0 / 0 |
| CMUMOSI | 0.7 | 0.01 | 686 | 67.2256 | 67.2256 | -3.36943e-05 | 0.000696895 | 0 | 0 / 0 |

Noise is applied to actual historical projected Values, not separately to Base/Gap. The same unit noise is reused at both sigmas.
Binary metrics/counts exclude y=0 and retain threshold prediction>0; MSE includes all valid labels.
The head-only intervention uses draw0; it is not a Monte Carlo estimate. Kappa is not task uncertainty.
AGGREGATE.json contains pattern, correctness, history, kappa-quartile and head strata. STATUS.json contains parity and covariance errors.
Full G matrices and per-draw predictions are retained in remote batch NPZ/ROWS artifacts, not committed to Git.
