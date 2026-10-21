# MOSI original Flat intermediate LR screen

INTERNAL TEST-ORACLE SCREEN

Seed66,150epochs,original GCNet missing; no Router/new block. Fresh initialization.
LR decays10x at epoch101. Each rate selects its own earliest TEST-best W-F1 epoch.
The eight-rate mean combines different checkpoints; not a single deployment score.

| Model | Mean ACC | Mean W-F1 | High ACC | High W-F1 | ΔMean W-F1(pp) |
|---|---:|---:|---:|---:|---:|
| original_1e-3 | 79.1159 | 79.1280 | 73.2724 | 73.3381 | +0.0000 |
| flat_0.0005 | 79.2492 | 79.1371 | 73.5772 | 73.5238 | +0.0091 |
| flat_0.0003 | 79.2683 | 79.2491 | 73.3740 | 73.4098 | +0.1211 |

| Model | Rate | Selected epoch | ACC | W-F1 |
|---|---|---:|---:|---:|---:|
| original_1e-3 | 0.0 | 70 | 86.7378 | 86.7346 |
| original_1e-3 | 0.1 | 103 | 84.2988 | 84.2326 |
| original_1e-3 | 0.2 | 103 | 82.3171 | 82.2008 |
| original_1e-3 | 0.3 | 105 | 82.1646 | 82.1220 |
| original_1e-3 | 0.4 | 84 | 77.5915 | 77.7199 |
| original_1e-3 | 0.5 | 62 | 75.4573 | 75.5949 |
| original_1e-3 | 0.6 | 66 | 74.8476 | 74.9072 |
| original_1e-3 | 0.7 | 54 | 69.5122 | 69.5122 |
| flat_0.0005 | 0.0 | 38 | 87.6524 | 87.5842 |
| flat_0.0005 | 0.1 | 38 | 83.8415 | 83.8940 |
| flat_0.0005 | 0.2 | 43 | 82.4695 | 82.4173 |
| flat_0.0005 | 0.3 | 37 | 80.9451 | 80.8996 |
| flat_0.0005 | 0.4 | 45 | 78.3537 | 77.7303 |
| flat_0.0005 | 0.5 | 55 | 77.4390 | 77.5327 |
| flat_0.0005 | 0.6 | 35 | 74.0854 | 73.8963 |
| flat_0.0005 | 0.7 | 117 | 69.2073 | 69.1425 |
| flat_0.0003 | 0.0 | 48 | 87.3476 | 87.2506 |
| flat_0.0003 | 0.1 | 52 | 83.9939 | 83.8712 |
| flat_0.0003 | 0.2 | 49 | 82.6220 | 82.5951 |
| flat_0.0003 | 0.3 | 50 | 80.7927 | 80.7412 |
| flat_0.0003 | 0.4 | 62 | 79.2683 | 79.3053 |
| flat_0.0003 | 0.5 | 48 | 76.0671 | 75.9958 |
| flat_0.0003 | 0.6 | 48 | 75.3049 | 75.3279 |
| flat_0.0003 | 0.7 | 51 | 68.7500 | 68.9057 |
