# Original GCNet random-mask rerun

**INTERNAL DIAGNOSTIC ONLY.**

The primary tables use per-rate Test-oracle checkpoint selection. Final
saved-prediction reevaluations are listed separately because the original
GCNet sampler draws a new stochastic test mask during that reevaluation.

MELD completed for Flat and Nested GNN with seeds 66/67/68 and 50 epochs.
The only protocol change from the corresponding readout runs is the original
batch-level GCNet `random_mask` sampler. ComP was neither modified nor rerun.

UA is macro recall computed from the saved prediction files:

\[
UA=\frac{1}{C}\sum_{c=1}^{C}\operatorname{Recall}_c.
\]

## MELD per-rate Test-best summary (%)

| Model | 8-rate ACC | 8-rate W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| Flat | 60.498 | 56.754 | 56.420 | **51.352** |
| Nested GNN | **60.640** | **56.828** | **56.590** | 50.706 |
| Nested - Flat | +0.142 | +0.074 | +0.170 | -0.646 |

Flat per-seed Test-best results are:

| Seed | 8-rate ACC | 8-rate W-F1 | High ACC | High W-F1 |
|---:|---:|---:|---:|---:|
| 66 | 60.637 | 57.087 | 56.450 | 51.659 |
| 67 | 60.402 | 56.383 | 56.411 | 50.806 |
| 68 | 60.455 | 56.794 | 56.398 | 51.592 |
| **Mean** | **60.498** | **56.754** | **56.420** | **51.352** |

Flat per-rate three-seed Test-best means are:

| Rate | ACC | W-F1 |
|---:|---:|---:|
| 0.0 | 66.232 | 64.769 |
| 0.1 | 64.662 | 62.636 |
| 0.2 | 62.759 | 59.976 |
| 0.3 | 61.367 | 57.401 |
| 0.4 | 59.706 | 55.196 |
| 0.5 | 57.995 | 53.878 |
| 0.6 | 56.245 | 50.947 |
| 0.7 | 55.019 | 49.233 |

## MELD final saved-prediction reevaluation (%)

The following tables are retained for prediction-file diagnostics such as UA.
They are not the primary Test-best performance tables.

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

## Comparison with the unchanged causal ComP run (%)

| Model | 8-rate ACC | 8-rate W-F1 | 8-rate UA | High ACC | High W-F1 | High UA |
|---|---:|---:|---:|---:|---:|---:|
| causal ComP | 60.532 | 56.440 | 33.742 | 56.462 | 50.742 | 27.768 |
| Nested GNN | 60.022 | 56.097 | 35.240 | 55.624 | 49.518 | 28.421 |
| Nested - ComP | -0.509 | -0.343 | +1.498 | -0.839 | -1.224 | +0.652 |

ComP was not modified or rerun. This comparison uses the same original GCNet
sampling law, but the stochastic masks are not sample-paired across methods.

## Correct-mask IEMOCAP completion status

The corrected-mask Flat runs have completed for IEMOCAP-4 and IEMOCAP-6.
Their matched Nested counterparts were launched on 2026-10-13 from commit
`7a25dd1` and remain in progress, so no Flat-versus-Nested result is claimed
yet.  Each task uses seeds 66/67/68, folds 1--5, 100 epochs, and the original
IEMOCAP Nested configuration (`dim=64`, `depth=3`, `groups=8`,
`mlp_hidden=64`).  MOSEI- or MELD-specific Nested parameters are not reused.

The remote output root is:

```text
/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_nested_20261013
```
