# MOSI memory/base/gap ablation

INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT

Protocol: cfg84 causal OSRAM (eta=.6, Flat, no-JEPA), trained from scratch; each seed/rate selects its own Test-oracle epoch.

| Variant | Rate | Mean W-F1 | SD |
|---|---:|---:|---:|
| local-only | 0.0 | 87.192 | 0.400 |
| local-only | 0.1 | 84.219 | 0.737 |
| local-only | 0.2 | 80.462 | 1.589 |
| local-only | 0.3 | 78.912 | 0.696 |
| local-only | 0.4 | 75.789 | 0.982 |
| local-only | 0.5 | 72.304 | 0.951 |
| local-only | 0.6 | 72.661 | 1.003 |
| local-only | 0.7 | 70.839 | 1.836 |
| local-base | 0.0 | 87.215 | 0.455 |
| local-base | 0.1 | 84.905 | 0.633 |
| local-base | 0.2 | 82.233 | 1.731 |
| local-base | 0.3 | 80.786 | 1.137 |
| local-base | 0.4 | 77.885 | 1.488 |
| local-base | 0.5 | 75.802 | 1.300 |
| local-base | 0.6 | 75.115 | 0.843 |
| local-base | 0.7 | 73.881 | 3.403 |
| raw-gap | 0.0 | 87.534 | 0.628 |
| raw-gap | 0.1 | 85.433 | 0.944 |
| raw-gap | 0.2 | 82.268 | 1.616 |
| raw-gap | 0.3 | 80.962 | 0.408 |
| raw-gap | 0.4 | 78.508 | 1.051 |
| raw-gap | 0.5 | 76.301 | 1.291 |
| raw-gap | 0.6 | 75.701 | 0.531 |
| raw-gap | 0.7 | 74.238 | 2.442 |
| full | 0.0 | 88.059 | 0.541 |
| full | 0.1 | 85.386 | 0.929 |
| full | 0.2 | 82.800 | 1.280 |
| full | 0.3 | 81.330 | 0.882 |
| full | 0.4 | 79.035 | 1.315 |
| full | 0.5 | 76.750 | 1.315 |
| full | 0.6 | 75.671 | 0.526 |
| full | 0.7 | 74.529 | 2.955 |

## Overall

| Variant | 8-rate mean | High-missing mean (.5/.6/.7) |
|---|---:|---:|
| local-only | 77.797 ± 0.303 | 71.935 ± 0.213 |
| local-base | 79.728 ± 0.414 | 74.933 ± 0.615 |
| raw-gap | 80.118 ± 0.496 | 75.413 ± 0.507 |
| full | 80.445 ± 0.433 | 75.650 ± 0.944 |
