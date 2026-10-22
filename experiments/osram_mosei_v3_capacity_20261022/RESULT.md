# MOSEI smaller Flat / wider V3, seed66

INTERNAL TEST-ORACLE SCREEN

Two fresh joint60epoch runs, Flat hidden1024, unchanged output1600/CHM8x64.
Wide V3: width256, PairMLP512, same four heads and original hierarchy.
Historical large Flat/default V3 are references, not fresh same-source training.
Only completed runs enter score tables. No inference is performed by reporting.

| Model | Scope | ACC | W-F1 | MSE |
|---|---|---:|---:|---:|
| large_flat | mean | 83.493 | 83.279 | 0.615713 |
| large_flat | high | 80.517 | 80.182 | 0.730888 |
| original_v3 | mean | 83.562 | 83.394 | 0.620059 |
| original_v3 | high | 80.279 | 80.121 | 0.739253 |
| small_flat | mean | 83.235 | 83.078 | 0.622369 |
| small_flat | high | 80.123 | 79.850 | 0.728433 |
| wide_v3 | mean | 83.286 | 83.104 | 0.630706 |
| wide_v3 | high | 80.095 | 79.820 | 0.744262 |

## State

- large_flat: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 10190.855628709018, 'peak_memory_mib': 1797.2666015625, 'parameters': None}
- original_v3: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 19543.178089867986, 'peak_memory_mib': 5561.05615234375, 'parameters': None}
- small_flat: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 6149.826286859985, 'peak_memory_mib': 1744.56787109375, 'parameters': {'flat_adapter': 6106176, 'total': 10080865, 'v3': 0}}
- wide_v3: {'status': 'complete', 'epoch': 60, 'error': None, 'seconds': 8434.529527131002, 'peak_memory_mib': 5548.6826171875, 'parameters': {'flat_adapter': 6106176, 'total': 12397503, 'v3': 2316638}}

## Per rate

| Model | Rate | Epoch | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|---:|
| large_flat | 0.0 | 22 | 87.369 | 87.265 | 0.463457 |
| large_flat | 0.1 | 27 | 86.076 | 85.957 | 0.526386 |
| large_flat | 0.2 | 19 | 85.416 | 85.269 | 0.531313 |
| large_flat | 0.3 | 19 | 84.700 | 84.540 | 0.559403 |
| large_flat | 0.4 | 27 | 82.829 | 82.656 | 0.652480 |
| large_flat | 0.5 | 16 | 82.058 | 81.766 | 0.680512 |
| large_flat | 0.6 | 33 | 80.407 | 80.039 | 0.735109 |
| large_flat | 0.7 | 27 | 79.086 | 78.741 | 0.777044 |
| original_v3 | 0.0 | 23 | 87.534 | 87.520 | 0.465421 |
| original_v3 | 0.1 | 23 | 86.214 | 86.162 | 0.506657 |
| original_v3 | 0.2 | 23 | 85.828 | 85.722 | 0.535726 |
| original_v3 | 0.3 | 31 | 84.920 | 84.754 | 0.579751 |
| original_v3 | 0.4 | 14 | 83.159 | 82.630 | 0.655160 |
| original_v3 | 0.5 | 26 | 82.003 | 81.850 | 0.725154 |
| original_v3 | 0.6 | 28 | 80.242 | 79.983 | 0.711228 |
| original_v3 | 0.7 | 32 | 78.591 | 78.530 | 0.781378 |
| small_flat | 0.0 | 23 | 87.149 | 87.175 | 0.470178 |
| small_flat | 0.1 | 21 | 86.103 | 85.911 | 0.503036 |
| small_flat | 0.2 | 23 | 85.388 | 85.352 | 0.541264 |
| small_flat | 0.3 | 23 | 84.645 | 84.581 | 0.578830 |
| small_flat | 0.4 | 17 | 82.223 | 82.055 | 0.700346 |
| small_flat | 0.5 | 45 | 81.866 | 81.556 | 0.691014 |
| small_flat | 0.6 | 23 | 79.857 | 79.622 | 0.727204 |
| small_flat | 0.7 | 24 | 78.646 | 78.371 | 0.767079 |
| wide_v3 | 0.0 | 25 | 87.314 | 87.238 | 0.476151 |
| wide_v3 | 0.1 | 25 | 85.856 | 85.716 | 0.516464 |
| wide_v3 | 0.2 | 25 | 85.360 | 85.181 | 0.546835 |
| wide_v3 | 0.3 | 28 | 84.755 | 84.686 | 0.587858 |
| wide_v3 | 0.4 | 32 | 82.719 | 82.555 | 0.685551 |
| wide_v3 | 0.5 | 18 | 82.031 | 81.929 | 0.685934 |
| wide_v3 | 0.6 | 21 | 79.499 | 79.254 | 0.754853 |
| wide_v3 | 0.7 | 28 | 78.756 | 78.276 | 0.791999 |

No multi-seed/independent-generalization claim. Per-rate Test-best may select different epochs.
Shrinking Flat alone and adding wider V3 are distinguished by the new small_flat control.
