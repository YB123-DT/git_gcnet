# V3 connection screen: MOSI / MOSEI seed66

INTERNAL TEST-ORACLE SCREEN

Three independent variants: head-preserving, source-slots, flat-hidden. Six formal runs only.
MOSI: 150 epochs, 3e-4 for 100 epochs then 3e-5 for 50. MOSEI: 60 epochs at 1e-3.
Original Flat1600, cyclic GCNet random masks 0.0–0.7, batch32, Adam, WD1e-5 and MSE.
Existing same-seed Flat results are reused; no new baseline training.

| Dataset | Model | Scope | ACC | W-F1 | ΔW-F1 vs Flat | MSE |
|---|---|---|---:|---:|---:|---:|
| CMUMOSI | Flat66 reference | mean | 79.268 | 79.249 | +0.000 | 1.494741 |
| CMUMOSI | Flat66 reference | high | 73.374 | 73.410 | +0.000 | 1.849947 |
| CMUMOSI | head-preserving | mean | 79.345 | 79.331 | +0.082 | 1.526487 |
| CMUMOSI | head-preserving | high | 73.374 | 73.438 | +0.029 | 1.827755 |
| CMUMOSI | source-slots | mean | 79.306 | 79.213 | -0.036 | 1.461002 |
| CMUMOSI | source-slots | high | 73.425 | 73.228 | -0.182 | 1.815032 |
| CMUMOSI | flat-hidden | mean | 79.097 | 79.112 | -0.137 | 1.499840 |
| CMUMOSI | flat-hidden | high | 72.612 | 72.725 | -0.685 | 1.852473 |
| CMUMOSEI | Flat66 reference | mean | 83.493 | 83.279 | +0.000 | 0.615713 |
| CMUMOSEI | Flat66 reference | high | 80.517 | 80.182 | +0.000 | 0.730888 |
| CMUMOSEI | head-preserving | mean | 83.476 | 83.293 | +0.014 | 0.611680 |
| CMUMOSEI | head-preserving | high | 80.435 | 80.188 | +0.006 | 0.721558 |
| CMUMOSEI | source-slots | mean | 83.328 | 83.172 | -0.107 | 0.636359 |
| CMUMOSEI | source-slots | high | 79.967 | 79.655 | -0.526 | 0.747915 |
| CMUMOSEI | flat-hidden | mean | 83.297 | 83.166 | -0.114 | 0.635765 |
| CMUMOSEI | flat-hidden | high | 80.050 | 79.802 | -0.380 | 0.742719 |

## Run state

- CMUMOSI_head-preserving: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 2515.3410836849944, 'parameters': {'flat_adapter': 9535104, 'total': 13845311, 'v3': 335518}, 'peak_memory_mib': 15456.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
- CMUMOSEI_head-preserving: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 8052.016561705037, 'parameters': {'flat_adapter': 9535104, 'total': 13845311, 'v3': 335518}, 'peak_memory_mib': 8894.0, 'gpu_uuid': 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'}
- CMUMOSI_source-slots: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 2449.0787579449825, 'parameters': {'flat_adapter': 9535104, 'total': 14024511, 'v3': 514718}, 'peak_memory_mib': 4654.0, 'gpu_uuid': 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'}
- CMUMOSEI_source-slots: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 8818.737218382012, 'parameters': {'flat_adapter': 9535104, 'total': 14024511, 'v3': 514718}, 'peak_memory_mib': 5182.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
- CMUMOSI_flat-hidden: {'status': 'complete', 'epoch': 150, 'error': None, 'seconds': 2475.3897302380065, 'parameters': {'flat_adapter': 9535104, 'total': 14121791, 'v3': 611998}, 'peak_memory_mib': 4624.0, 'gpu_uuid': 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'}
- CMUMOSEI_flat-hidden: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 8810.361422343005, 'parameters': {'flat_adapter': 9535104, 'total': 14121791, 'v3': 611998}, 'peak_memory_mib': 6182.0, 'gpu_uuid': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}

Only completed formal runs appear in score tables. Smoke checks are not performance evidence.
The eight rates may select different epochs by Test-best W-F1. Their mean is an internal
Test-oracle statistic, not one checkpoint deployed across all rates. No extra seeds or HPO.
Time and GPU memory are measured under shared-resource scheduling; no isolated speed claim.
