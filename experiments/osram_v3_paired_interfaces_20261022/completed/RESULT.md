# V3 paired interfaces: MOSI seed66

INTERNAL TEST-ORACLE SCREEN

Three pre-registered A/B pairs: topology, Local Skip input, task outlet normalization.
MOSI only: 150 epochs, 3e-4 for 100 epochs then 3e-5 for 50; six formal runs.
Original Flat1600, cyclic GCNet random masks 0.0–0.7, batch32, Adam, WD1e-5 and MSE.
Existing same-seed Flat results are reused; no new baseline training.

| Dataset | Model | Scope | ACC | W-F1 | ΔW-F1 vs Flat | MSE |
|---|---|---|---:|---:|---:|---:|
| CMUMOSI | Flat66 reference | mean | 79.268 | 79.249 | +0.000 | 1.494741 |
| CMUMOSI | Flat66 reference | high | 73.374 | 73.410 | +0.000 | 1.849947 |
| CMUMOSI | topology-full | mean | 79.306 | 79.139 | -0.110 | 1.460403 |
| CMUMOSI | topology-full | high | 74.035 | 73.713 | +0.303 | 1.766467 |
| CMUMOSI | topology-diagonal | mean | 78.697 | 78.617 | -0.632 | 1.506573 |
| CMUMOSI | topology-diagonal | high | 73.018 | 72.976 | -0.434 | 1.855659 |
| CMUMOSI | local-adapter | mean | 79.421 | 79.323 | +0.074 | 1.513370 |
| CMUMOSI | local-adapter | high | 73.730 | 73.551 | +0.141 | 1.846165 |
| CMUMOSI | local-shared | mean | 79.002 | 78.871 | -0.378 | 1.466736 |
| CMUMOSI | local-shared | high | 73.069 | 72.779 | -0.631 | 1.794319 |
| CMUMOSI | task-pre-norm | mean | 78.868 | 78.860 | -0.389 | 1.508757 |
| CMUMOSI | task-pre-norm | high | 73.120 | 73.156 | -0.253 | 1.877001 |
| CMUMOSI | task-post-norm | mean | 78.982 | 78.806 | -0.443 | 1.515247 |
| CMUMOSI | task-post-norm | high | 72.917 | 72.581 | -0.828 | 1.817520 |

## Paired effects: B minus A

| Pair | Scope | ΔACC | ΔW-F1 | ΔMSE |
|---|---|---:|---:|---:|
| topology-full / topology-diagonal | high | -1.016 | -0.737 | +0.089192 |
| topology-full / topology-diagonal | mean | -0.610 | -0.523 | +0.046170 |
| local-adapter / local-shared | high | -0.661 | -0.772 | -0.051846 |
| local-adapter / local-shared | mean | -0.419 | -0.452 | -0.046634 |
| task-pre-norm / task-post-norm | high | -0.203 | -0.575 | -0.059482 |
| task-pre-norm / task-post-norm | mean | +0.114 | -0.054 | +0.006490 |

## Run state

- CMUMOSI_topology-full: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3947.2328920220025, 'parameters': {'flat_adapter': 9535104, 'total': 13844799, 'v3': 335006}, 'peak_memory_mib': 4700.0, 'gpu_uuid': 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}
- CMUMOSI_topology-diagonal: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3747.233645988046, 'parameters': {'flat_adapter': 9535104, 'total': 13844799, 'v3': 335006}, 'peak_memory_mib': 4700.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
- CMUMOSI_local-adapter: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3673.9208674099646, 'parameters': {'flat_adapter': 9535104, 'total': 13777471, 'v3': 267678}, 'peak_memory_mib': 4664.0, 'gpu_uuid': 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}
- CMUMOSI_local-shared: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3842.5353555949987, 'parameters': {'flat_adapter': 9535104, 'total': 13777471, 'v3': 267678}, 'peak_memory_mib': 4664.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
- CMUMOSI_task-pre-norm: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3798.063852899999, 'parameters': {'flat_adapter': 9535104, 'total': 14122879, 'v3': 613086}, 'peak_memory_mib': 4676.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
- CMUMOSI_task-post-norm: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 3898.155736047018, 'parameters': {'flat_adapter': 9535104, 'total': 14122879, 'v3': 613086}, 'peak_memory_mib': 4662.0, 'gpu_uuid': 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}

Only completed formal runs appear in score tables. Smoke checks are not performance evidence.
The eight rates may select different epochs by Test-best W-F1. Their mean is an internal
Test-oracle statistic, not one checkpoint deployed across all rates. No extra seeds or HPO.
Time and GPU memory are measured under shared-resource scheduling; no isolated speed claim.
