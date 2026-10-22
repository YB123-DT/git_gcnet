# MOSI direct V3 read inputs

INTERNAL TEST-ORACLE SCREEN

Seed66,100epochs,3e-4,original GCNet masks. Only completed runs enter scores.
Historical150-epoch controls are reselected from their FIRST100 epoch logs.

| Model | ACC | W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| original_large_flat | 79.268 | 79.249 | 73.374 | 73.410 |
| flat512 | 78.944 | 78.882 | 72.967 | 72.949 |
| flat512_v3_residual | 79.497 | 79.453 | 73.476 | 73.519 |

State: {'status': 'preflight', 'epoch': None, 'error': None, 'seconds': None, 'parameters': None}

Direct uses random decoders; residual uses zero decoders.
This is not an addition-only causal comparison. Internal V3 residuals remain.
No CHM/LocalSkip changes. No additional reference training/inference.

| Model | Rate | Epoch | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|---:|
| original_large_flat | 0.0 | 48 | 87.348 | 87.251 | 1.018859 |
| original_large_flat | 0.1 | 52 | 83.994 | 83.871 | 1.204979 |
| original_large_flat | 0.2 | 49 | 82.622 | 82.595 | 1.342931 |
| original_large_flat | 0.3 | 50 | 80.793 | 80.741 | 1.306329 |
| original_large_flat | 0.4 | 62 | 79.268 | 79.305 | 1.534990 |
| original_large_flat | 0.5 | 48 | 76.067 | 75.996 | 1.579528 |
| original_large_flat | 0.6 | 48 | 75.305 | 75.328 | 1.828822 |
| original_large_flat | 0.7 | 51 | 68.750 | 68.906 | 2.141490 |
| flat512 | 0.0 | 28 | 87.652 | 87.643 | 1.012037 |
| flat512 | 0.1 | 59 | 83.841 | 83.768 | 1.182274 |
| flat512 | 0.2 | 53 | 82.317 | 82.334 | 1.333773 |
| flat512 | 0.3 | 55 | 81.098 | 80.853 | 1.321007 |
| flat512 | 0.4 | 55 | 77.744 | 77.613 | 1.546923 |
| flat512 | 0.5 | 52 | 75.762 | 75.896 | 1.557490 |
| flat512 | 0.6 | 55 | 73.628 | 73.343 | 1.784629 |
| flat512 | 0.7 | 37 | 69.512 | 69.610 | 2.098098 |
| flat512_v3_residual | 0.0 | 26 | 87.195 | 87.201 | 1.031947 |
| flat512_v3_residual | 0.1 | 28 | 84.604 | 84.450 | 1.158018 |
| flat512_v3_residual | 0.2 | 44 | 82.774 | 82.629 | 1.375852 |
| flat512_v3_residual | 0.3 | 28 | 82.165 | 82.041 | 1.335013 |
| flat512_v3_residual | 0.4 | 28 | 78.811 | 78.748 | 1.470648 |
| flat512_v3_residual | 0.5 | 42 | 76.067 | 76.061 | 1.517919 |
| flat512_v3_residual | 0.6 | 42 | 74.238 | 74.308 | 1.841362 |
| flat512_v3_residual | 0.7 | 46 | 70.122 | 70.187 | 1.933084 |
