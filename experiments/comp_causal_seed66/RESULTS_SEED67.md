# Causal ComP CMU-MOSI Seed-67 Results

## Completion evidence

- Seed: 67
- Missing rates: `0.0` through `0.7`
- Training budget: 300 epochs per rate, Stage 1/Stage 2 split at epoch 150
- Executed repository commit: `80c1b3656619327e0a101fcc9eb9c213beddc910`
- Upstream ComP commit: `28192d3a5683543d7383e40898f9a98d1f114a08`
- Output root: `/data2/yb/paper/05_reproduction/runs/ComP_causal/seed67/CMUMOSI`

All eight logs contain epoch 299, all result files contain `Folder avg:`, all manifests match seed 67 and the executed commit, and the error scan found no traceback, CUDA error, OOM, or NaN.

## Seed-67 matched comparison

The values retain the upstream test-set peak-selection protocol. The non-causal comparison is the completed CMU-MOSI seed-67 sweep with the same features and hyperparameters.

| Missing rate | Causal best epoch | Causal ACC (%) | Causal F1 (%) | Non-causal ACC (%) | Non-causal F1 (%) | ΔACC (pp) | ΔF1 (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 207 | 87.043 | 87.026 | 87.043 | 86.996 | +0.000 | +0.030 |
| 0.1 | 208 | 85.366 | 85.403 | 86.128 | 86.150 | -0.762 | -0.747 |
| 0.2 | 205 | 83.689 | 83.750 | 84.146 | 84.193 | -0.457 | -0.443 |
| 0.3 | 182 | 81.860 | 81.916 | 83.232 | 83.247 | -1.372 | -1.332 |
| 0.4 | 214 | 79.421 | 79.546 | 80.488 | 80.538 | -1.067 | -0.992 |
| 0.5 | 223 | 76.524 | 76.584 | 77.591 | 77.636 | -1.067 | -1.052 |
| 0.6 | 210 | 75.000 | 74.853 | 74.390 | 74.390 | +0.610 | +0.463 |
| 0.7 | 227 | 73.323 | 73.365 | 73.476 | 73.544 | -0.152 | -0.178 |
| **8-rate mean** | — | **80.278** | **80.305** | **80.812** | **80.837** | **-0.534** | **-0.531** |

## Two-seed causal summary

Values below are mean ± sample standard deviation across seeds 66 and 67. Each seed first contributes one official result at each missing rate.

| Missing rate | Causal ACC (%) | Causal F1 (%) |
|---:|---:|---:|
| 0.0 | 86.890 ± 0.216 | 86.870 ± 0.221 |
| 0.1 | 85.518 ± 0.216 | 85.547 ± 0.203 |
| 0.2 | 83.918 ± 0.323 | 83.940 ± 0.269 |
| 0.3 | 81.936 ± 0.108 | 81.917 ± 0.002 |
| 0.4 | 80.259 ± 1.186 | 80.322 ± 1.097 |
| 0.5 | 77.058 ± 0.755 | 77.141 ± 0.788 |
| 0.6 | 74.390 ± 0.862 | 74.364 ± 0.692 |
| 0.7 | 73.171 ± 0.216 | 73.264 ± 0.143 |

Across each seed's eight-rate mean:

| Model | ACC (%) | F1 (%) |
|---|---:|---:|
| Causal ComP | **80.393 ± 0.162** | **80.421 ± 0.163** |
| Non-causal ComP | 80.793 ± 0.027 | 80.821 ± 0.022 |
| Paired causal delta | **-0.400 ± 0.189 pp** | **-0.400 ± 0.185 pp** |

The second seed confirms that the causal model remains close to the original but suggests a small, consistent average reduction: the matched causal delta is negative for both seeds. With only two seeds, these standard deviations are descriptive; no significance claim is justified. Rate 0.4 shows the largest cross-seed variability, while rates 0.0, 0.1, 0.3, and 0.7 are comparatively stable.
