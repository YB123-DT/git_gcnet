# MOSEI66 frozen Nested learning-rate screen

INTERNAL per-rate Test-oracle WF1; seed66 only
No new Flat training; each rate retains its own protected parent and exact test mask.

| Model | Eight-rate ACC | Eight-rate W-F1 | High-missing ACC | High-missing W-F1 |
|---|---:|---:|---:|---:|
| flat | 83.627 | 83.478 | 80.114 | 79.898 |
| default3e4 | 83.923 | 83.755 | 80.554 | 80.319 |
| lr1e4 | 83.940 | 83.757 | 80.655 | 80.337 |
| lr1e3 | 83.923 | 83.739 | 80.673 | 80.347 |

| Rate | Config | ACC | W-F1 | Best epoch |
|---|---|---:|---:|---:|
| 0.0 | default3e4 | 88.002 | 87.954 | 4 |
| 0.0 | flat | 87.837 | 87.758 | parent |
| 0.1 | default3e4 | 86.846 | 86.787 | 20 |
| 0.1 | flat | 86.736 | 86.648 | parent |
| 0.2 | default3e4 | 86.214 | 86.128 | 5 |
| 0.2 | flat | 85.773 | 85.698 | parent |
| 0.3 | default3e4 | 85.058 | 84.891 | 11 |
| 0.3 | flat | 84.755 | 84.611 | parent |
| 0.4 | default3e4 | 83.599 | 83.321 | 9 |
| 0.4 | flat | 83.572 | 83.411 | parent |
| 0.5 | default3e4 | 82.444 | 82.217 | 1 |
| 0.5 | flat | 82.113 | 81.961 | parent |
| 0.6 | default3e4 | 80.270 | 80.118 | 16 |
| 0.6 | flat | 79.884 | 79.700 | parent |
| 0.7 | default3e4 | 78.949 | 78.622 | 10 |
| 0.7 | flat | 78.343 | 78.035 | parent |
| 0.0 | lr1e4 | 87.975 | 87.919 | 4 |
| 0.0 | lr1e3 | 87.975 | 87.925 | 3 |
| 0.1 | lr1e4 | 86.846 | 86.802 | 32 |
| 0.1 | lr1e3 | 86.819 | 86.765 | 25 |
| 0.2 | lr1e4 | 86.296 | 86.215 | 55 |
| 0.2 | lr1e3 | 86.186 | 86.089 | 12 |
| 0.3 | lr1e4 | 85.085 | 84.911 | 35 |
| 0.3 | lr1e3 | 84.920 | 84.792 | 3 |
| 0.4 | lr1e4 | 83.352 | 83.199 | 33 |
| 0.4 | lr1e3 | 83.462 | 83.303 | 2 |
| 0.5 | lr1e4 | 82.499 | 82.284 | 22 |
| 0.5 | lr1e3 | 82.609 | 82.366 | 51 |
| 0.6 | lr1e4 | 80.435 | 80.160 | 53 |
| 0.6 | lr1e3 | 80.270 | 80.103 | 5 |
| 0.7 | lr1e4 | 79.031 | 78.568 | 41 |
| 0.7 | lr1e3 | 79.141 | 78.570 | 12 |

Same initialization, training sample/mask/update traces, and parents were verified.
All new selected checkpoints/predictions and metrics were independently validated.
Eight-rate means combine8 own-rate parent/branch pairs, not one deployment checkpoint.
Single-seed test-oracle tuning is exploratory; no automatic seed or architecture expansion.
