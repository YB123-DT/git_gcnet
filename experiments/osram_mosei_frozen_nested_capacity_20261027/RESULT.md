# MOSEI66 frozen Nested capacity screen

INTERNAL per-rate Test-oracle WF1; seed66 only

| Variant | Added params | Eight-rate ACC | Eight-rate W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|---:|
| flat | 0 | 83.627 | 83.478 | 80.114 | 79.898 |
| default3e4 | 159235 | 83.923 | 83.755 | 80.554 | 80.319 |
| dim32 | 80163 | 84.005 | 83.835 | 80.682 | 80.420 |
| mlp128 | 216899 | 83.943 | 83.780 | 80.627 | 80.365 |
| depth2 | 146690 | 83.974 | 83.797 | 80.682 | 80.373 |

| Rate | Config | ACC | W-F1 | Selected epoch |
|---|---|---:|---:|---:|
| 0.0 | default3e4 | 88.002 | 87.954 | 4 |
| 0.0 | flat | 87.837 | 87.758 | 22 |
| 0.1 | default3e4 | 86.846 | 86.787 | 20 |
| 0.1 | flat | 86.736 | 86.648 | 22 |
| 0.2 | default3e4 | 86.214 | 86.128 | 5 |
| 0.2 | flat | 85.773 | 85.698 | 12 |
| 0.3 | default3e4 | 85.058 | 84.891 | 11 |
| 0.3 | flat | 84.755 | 84.611 | 17 |
| 0.4 | default3e4 | 83.599 | 83.321 | 9 |
| 0.4 | flat | 83.572 | 83.411 | 12 |
| 0.5 | default3e4 | 82.444 | 82.217 | 1 |
| 0.5 | flat | 82.113 | 81.961 | 16 |
| 0.6 | default3e4 | 80.270 | 80.118 | 16 |
| 0.6 | flat | 79.884 | 79.700 | 16 |
| 0.7 | default3e4 | 78.949 | 78.622 | 10 |
| 0.7 | flat | 78.343 | 78.035 | 23 |
| 0.0 | dim32 | 88.057 | 88.011 | 4 |
| 0.0 | mlp128 | 88.002 | 87.935 | 15 |
| 0.0 | depth2 | 88.057 | 88.015 | 4 |
| 0.1 | dim32 | 86.957 | 86.895 | 20 |
| 0.1 | mlp128 | 86.874 | 86.811 | 16 |
| 0.1 | depth2 | 86.901 | 86.853 | 16 |
| 0.2 | dim32 | 86.186 | 86.116 | 15 |
| 0.2 | mlp128 | 86.241 | 86.149 | 5 |
| 0.2 | depth2 | 86.269 | 86.186 | 5 |
| 0.3 | dim32 | 85.085 | 84.926 | 2 |
| 0.3 | mlp128 | 85.140 | 84.952 | 13 |
| 0.3 | depth2 | 85.085 | 84.920 | 13 |
| 0.4 | dim32 | 83.709 | 83.468 | 9 |
| 0.4 | mlp128 | 83.407 | 83.298 | 27 |
| 0.4 | depth2 | 83.434 | 83.287 | 53 |
| 0.5 | dim32 | 82.334 | 82.235 | 8 |
| 0.5 | mlp128 | 82.416 | 82.268 | 16 |
| 0.5 | depth2 | 82.334 | 82.238 | 8 |
| 0.6 | dim32 | 80.325 | 80.129 | 23 |
| 0.6 | mlp128 | 80.297 | 80.140 | 16 |
| 0.6 | depth2 | 80.407 | 80.206 | 23 |
| 0.7 | dim32 | 79.389 | 78.896 | 15 |
| 0.7 | mlp128 | 79.169 | 78.686 | 12 |
| 0.7 | depth2 | 79.307 | 78.674 | 15 |

Only Nested architecture changes; CHM,Flat,mask and training budget are fixed.
dim32 explicitly keeps MLP64; registered plain dim32 would also shrink MLP and is not used.
Initial logits equal the protected parent, but changed-shape parameter tensors are not claimed identical to default.
Default-dimension Sweep and original Nested were verified numerically equivalent with opened decoders.
Eight-rate means combine8 own-rate parent/branch pairs. No independent generalization or significance claim.
