# Causal ComP CMU-MOSI Seed-66 Results

## Completion evidence

- Dataset: CMU-MOSI
- Seed: 66
- Missing rates: `0.0` through `0.7`
- Training budget: 300 epochs per rate, Stage 1/Stage 2 split at epoch 150
- Executed repository commit: `ce5b0b45a8d941e92b2e178f54a9fd93600aa45c`
- Upstream ComP commit: `28192d3a5683543d7383e40898f9a98d1f114a08`
- Host/device: `user23`, physical GPU 3, NVIDIA TITAN Xp
- Environment: `comp-repro`, Python 3.8.20, PyTorch 1.12.0+cu113
- Output root: `/data2/yb/paper/05_reproduction/runs/ComP_causal/seed66/CMUMOSI`

All eight logs contain epoch 299, all eight result files contain `Folder avg:`, all manifests match the executed commit and protocol, and the error scan found no traceback, CUDA error, OOM, or NaN. Four concurrent workers reached an observed aggregate GPU allocation of approximately 7.9 GiB.

## Official-protocol results

These values follow the upstream protocol, which selects the best epoch by test-set F1. The comparison uses the completed non-causal seed-66 results under the same features, environment, hyperparameters, and missing rates.

| Missing rate | Causal best epoch | Causal ACC (%) | Causal F1 (%) | Non-causal ACC (%) | Non-causal F1 (%) | ΔACC (pp) | ΔF1 (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 185 | 86.738 | 86.714 | 87.348 | 87.317 | -0.610 | -0.604 |
| 0.1 | 189 | 85.671 | 85.690 | 85.976 | 85.995 | -0.305 | -0.304 |
| 0.2 | 217 | 84.146 | 84.130 | 83.994 | 83.956 | +0.152 | +0.175 |
| 0.3 | 232 | 82.012 | 81.919 | 82.012 | 81.994 | +0.000 | -0.075 |
| 0.4 | 186 | 81.098 | 81.098 | 80.335 | 80.429 | +0.762 | +0.668 |
| 0.5 | 267 | 77.591 | 77.698 | 79.268 | 79.297 | -1.677 | -1.598 |
| 0.6 | 259 | 73.780 | 73.874 | 73.933 | 73.973 | -0.152 | -0.099 |
| 0.7 | 208 | 73.018 | 73.163 | 73.323 | 73.480 | -0.305 | -0.317 |
| **8-rate mean** | — | **80.507** | **80.536** | **80.774** | **80.805** | **-0.267** | **-0.269** |

The causal variant is close to the non-causal baseline on this seed: the mean reduction is 0.267 percentage points in ACC and 0.269 points in F1. It improves at rates 0.2 and 0.4, ties ACC at 0.3, and has its largest reduction at rate 0.5.

## Additional metrics

| Missing rate | Causal MAE | Causal Corr | Non-causal MAE | Non-causal Corr |
|---:|---:|---:|---:|---:|
| 0.0 | 0.738153 | 0.819200 | 0.744586 | 0.836217 |
| 0.1 | 0.779326 | 0.792811 | 0.784532 | 0.806252 |
| 0.2 | 0.803850 | 0.776496 | 0.841100 | 0.763969 |
| 0.3 | 0.883322 | 0.718346 | 0.874915 | 0.748929 |
| 0.4 | 0.958805 | 0.702573 | 0.931431 | 0.715985 |
| 0.5 | 1.038975 | 0.635103 | 1.016833 | 0.674579 |
| 0.6 | 1.065383 | 0.597437 | 1.115806 | 0.632090 |
| 0.7 | 1.175245 | 0.525785 | 1.205871 | 0.555894 |
| **8-rate mean** | **0.930382** | **0.695969** | **0.939384** | **0.716739** |

## Interpretation limits

This is one random seed. Missing rates are different experimental conditions, not independent seed replications, so their mean does not provide a standard deviation or a valid significance test. The causal-versus-non-causal difference should therefore be described as a seed-66 matched comparison, not a statistically established population effect. The retained upstream test-peak selection also makes these values directly comparable to the reproduction but unsuitable as an unbiased validation-selected estimate.
