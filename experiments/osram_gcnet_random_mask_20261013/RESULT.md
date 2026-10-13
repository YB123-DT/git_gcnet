# Original GCNet random-mask rerun

**INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle checkpoint selection.**

MELD completed for Flat and Nested GNN with seeds 66/67/68 and 50 epochs.
The only protocol change from the corresponding readout runs is the original
batch-level GCNet `random_mask` sampler. ComP was neither modified nor rerun.

UA is macro recall computed from the saved prediction files:

\[
UA=\frac{1}{C}\sum_{c=1}^{C}\operatorname{Recall}_c.
\]

## MELD three-seed summary (%)

| Model | 8-rate ACC | 8-rate W-F1 | 8-rate UA | High ACC | High W-F1 | High UA |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 59.746 | 55.892 | 34.980 | 55.215 | 49.930 | 29.213 |
| Nested GNN | **60.022** | **56.097** | **35.240** | **55.624** | 49.518 | 28.421 |
| Nested - Flat | +0.276 | +0.205 | +0.260 | +0.409 | -0.412 | -0.792 |

`High` is the mean over missing rates 0.5/0.6/0.7.

## MELD per-rate three-seed means (%)

| Rate | Flat ACC | Flat W-F1 | Flat UA | Nested ACC | Nested W-F1 | Nested UA |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.232 | 64.769 | 44.903 | 66.360 | 65.064 | 45.678 |
| 0.1 | 63.934 | 61.849 | 41.607 | 64.483 | 62.193 | 41.493 |
| 0.2 | 62.171 | 59.240 | 38.597 | 62.439 | 59.572 | 38.166 |
| 0.3 | 60.319 | 56.208 | 33.822 | 60.473 | 57.394 | 36.346 |
| 0.4 | 59.668 | 55.279 | 33.272 | 59.553 | 55.997 | 34.975 |
| 0.5 | 56.603 | 52.246 | 31.808 | 56.794 | 51.805 | 31.243 |
| 0.6 | 54.968 | 49.247 | 27.632 | 54.981 | 48.783 | 27.818 |
| 0.7 | 54.074 | 48.296 | 28.199 | 55.096 | 47.967 | 26.201 |

The Nested readout improves the aggregate ACC, W-F1, and UA slightly. At high
missing rates it improves ACC but not W-F1 or UA; therefore the result does not
support a uniform high-missing improvement claim.

IEMOCAP-4/6 jobs remain in progress and are not reported here.
