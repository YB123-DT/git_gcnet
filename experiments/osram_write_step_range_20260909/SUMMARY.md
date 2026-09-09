# Frozen write-step range extension

INTERNAL DIAGNOSTIC ONLY. The extension (.6, .4, .2, 0) was chosen AFTER the previous four-point results; the combined eight points were not all preregistered. Inference only; no training, new mechanisms, or retrain gate.

Eta 0 disables memory context at frozen weights; it is not a trained Local model. Checkpoint/config/epoch/selection metadata and per-seed/per-rate masks match across inherited and new sources; weights-unchanged flags are true. This validates metadata, not an independent checkpoint-file rehash. Exact source paths and recorded hashes are in provenance.csv.

Wf1 is percent; differences are percentage points (pp). Mean ± sample SD uses five equally weighted seeds. `all` first averages five rates within each seed; rates and heads are not independent replicates. A sampled-mean peak and its nearest tested lower/higher eta only describe a sampled bracket, not a continuous optimum. Five seeds and adaptive extension do not support a significance or unbiased model-selection claim.

## iemocap4: weighted F1 (%)

| Rate | eta 1 | eta 0.95 | eta 0.9 | eta 0.8 | eta 0.6 | eta 0.4 | eta 0.2 | eta 0 |
|---|---|---|---|---|---|---|---|---|
| 0 | 84.334 ± 0.473 | 84.452 ± 0.452 | 84.471 ± 0.470 | 84.974 ± 0.359 | 85.181 ± 0.351 | 84.896 ± 0.619 | 83.794 ± 1.028 | 74.446 ± 1.462 |
| 0.1 | 83.924 ± 0.619 | 84.159 ± 0.527 | 84.209 ± 0.466 | 84.342 ± 0.344 | 84.651 ± 0.501 | 84.566 ± 0.491 | 83.521 ± 0.525 | 72.246 ± 0.771 |
| 0.3 | 81.993 ± 0.598 | 82.173 ± 0.557 | 82.193 ± 0.616 | 82.520 ± 0.672 | 82.539 ± 0.809 | 82.564 ± 0.619 | 81.354 ± 1.118 | 66.806 ± 1.876 |
| 0.5 | 80.118 ± 1.285 | 80.197 ± 1.242 | 80.339 ± 1.113 | 80.533 ± 1.108 | 80.737 ± 1.393 | 80.336 ± 1.268 | 78.681 ± 1.472 | 61.382 ± 2.209 |
| 0.7 | 77.750 ± 1.682 | 77.990 ± 1.645 | 78.122 ± 1.623 | 78.451 ± 1.510 | 78.682 ± 1.471 | 78.434 ± 1.284 | 76.219 ± 0.656 | 56.988 ± 1.556 |
| all | 81.624 ± 0.538 | 81.794 ± 0.534 | 81.867 ± 0.520 | 82.164 ± 0.514 | 82.358 ± 0.647 | 82.159 ± 0.487 | 80.714 ± 0.839 | 66.374 ± 1.026 |

Five-rate per-seed best sampled eta(s): seed 66: 0.6, seed 67: 0.8, seed 68: 0.4, seed 69: 0.6, seed 70: 0.6.

| Rate | Sampled peak eta | Lower / higher tested eta | Location | Peak−lower pp (positive seeds) | Peak−higher pp (positive seeds) |
|---|---|---|---|---|---|
| 0 | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.285 ± 0.293 (3/5) | 0.207 ± 0.225 (5/5) |
| 0.1 | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.085 ± 0.446 (3/5) | 0.309 ± 0.265 (5/5) |
| 0.3 | 0.4 | 0.2 / 0.6 | interior sampled peak | 1.210 ± 0.743 (5/5) | 0.025 ± 0.573 (2/5) |
| 0.5 | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.401 ± 0.315 (4/5) | 0.205 ± 0.374 (2/5) |
| 0.7 | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.248 ± 0.282 (4/5) | 0.231 ± 0.741 (3/5) |
| all | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.199 ± 0.259 (4/5) | 0.194 ± 0.283 (4/5) |
## mosi: weighted F1 (%)

| Rate | eta 1 | eta 0.95 | eta 0.9 | eta 0.8 | eta 0.6 | eta 0.4 | eta 0.2 | eta 0 |
|---|---|---|---|---|---|---|---|---|
| 0 | 86.786 ± 0.610 | 86.786 ± 0.610 | 86.755 ± 0.547 | 86.812 ± 0.519 | 86.875 ± 0.480 | 86.940 ± 0.481 | 87.068 ± 0.588 | 86.869 ± 0.893 |
| 0.1 | 84.797 ± 1.154 | 84.827 ± 1.187 | 84.707 ± 1.140 | 84.646 ± 1.168 | 84.831 ± 1.219 | 84.807 ± 1.264 | 84.844 ± 1.191 | 83.828 ± 1.403 |
| 0.3 | 79.620 ± 0.998 | 79.714 ± 0.982 | 79.751 ± 1.075 | 80.056 ± 0.950 | 80.229 ± 0.896 | 80.418 ± 1.182 | 79.943 ± 1.085 | 78.003 ± 0.742 |
| 0.5 | 76.322 ± 1.645 | 76.286 ± 1.690 | 76.273 ± 1.681 | 76.208 ± 1.528 | 76.309 ± 1.703 | 76.018 ± 1.991 | 75.616 ± 1.734 | 72.163 ± 0.856 |
| 0.7 | 72.806 ± 3.232 | 73.071 ± 3.167 | 73.236 ± 2.936 | 73.280 ± 2.675 | 73.506 ± 2.493 | 73.441 ± 2.231 | 72.430 ± 2.261 | 69.009 ± 3.903 |
| all | 80.066 ± 0.541 | 80.137 ± 0.506 | 80.145 ± 0.365 | 80.201 ± 0.341 | 80.350 ± 0.363 | 80.325 ± 0.392 | 79.980 ± 0.381 | 77.974 ± 0.975 |

Five-rate per-seed best sampled eta(s): seed 66: 0.6, seed 67: 0.6, seed 68: 0.4, seed 69: 1, seed 70: 0.95.

| Rate | Sampled peak eta | Lower / higher tested eta | Location | Peak−lower pp (positive seeds) | Peak−higher pp (positive seeds) |
|---|---|---|---|---|---|
| 0 | 0.2 | 0.0 / 0.4 | interior sampled peak | 0.199 ± 0.794 (2/5) | 0.128 ± 0.121 (5/5) |
| 0.1 | 0.2 | 0.0 / 0.4 | interior sampled peak | 1.016 ± 0.342 (5/5) | 0.037 ± 0.222 (2/5) |
| 0.3 | 0.4 | 0.2 / 0.6 | interior sampled peak | 0.475 ± 0.800 (4/5) | 0.189 ± 0.544 (3/5) |
| 0.5 | 1 | 0.95 /  | upper tested boundary; no interior optimum established | 0.036 ± 0.309 (3/5) | N/A |
| 0.7 | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.065 ± 0.577 (3/5) | 0.226 ± 0.205 (4/5) |
| all | 0.6 | 0.4 / 0.8 | interior sampled peak | 0.025 ± 0.222 (3/5) | 0.149 ± 0.137 (4/5) |

## Cross-dataset comparison

The datasets agree on the five-rate sampled-mean peak set.
Peak sets: iemocap4: [0.6]; mosi: [0.6].

## Old-read and new-write errors

E_old is actual-read retention.err_decay (NO_HISTORY excluded), not err_post. E_new is observed current_observed_write_fit.err_after. Modalities remain separate. Undefined retention at rate zero is omitted, never zero-filled; no five-rate aggregate is fabricated when a rate is undefined. Run-level means are equally weighted across seeds, not weighted by token counts.

| Dataset | Rate | Mode | Error | Modality | Mean ± SD |
|---|---|---|---|---|---|
| iemocap4 | 0 | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | 0 | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | 0 | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | 0 | fixed0.2 | err_after | audio | 0.518562 ± 0.0269336 |
| iemocap4 | 0 | fixed0.2 | err_after | text | 0.719643 ± 0.0127017 |
| iemocap4 | 0 | fixed0.2 | err_after | visual | 0.528436 ± 0.0176541 |
| iemocap4 | 0 | fixed0.4 | err_after | audio | 0.377865 ± 0.021527 |
| iemocap4 | 0 | fixed0.4 | err_after | text | 0.55086 ± 0.0111537 |
| iemocap4 | 0 | fixed0.4 | err_after | visual | 0.371907 ± 0.015492 |
| iemocap4 | 0 | fixed0.6 | err_after | audio | 0.257455 ± 0.0151774 |
| iemocap4 | 0 | fixed0.6 | err_after | text | 0.383039 ± 0.0083466 |
| iemocap4 | 0 | fixed0.6 | err_after | visual | 0.247711 ± 0.0117857 |
| iemocap4 | 0 | fixed0.8 | err_after | audio | 0.135542 ± 0.00814309 |
| iemocap4 | 0 | fixed0.8 | err_after | text | 0.203706 ± 0.00459438 |
| iemocap4 | 0 | fixed0.8 | err_after | visual | 0.128743 ± 0.00669031 |
| iemocap4 | 0 | fixed0.9 | err_after | audio | 0.0707107 ± 0.00426932 |
| iemocap4 | 0 | fixed0.9 | err_after | text | 0.106549 ± 0.00242235 |
| iemocap4 | 0 | fixed0.9 | err_after | visual | 0.0669382 ± 0.00358426 |
| iemocap4 | 0 | fixed0.95 | err_after | audio | 0.0367719 ± 0.0022213 |
| iemocap4 | 0 | fixed0.95 | err_after | text | 0.0554805 ± 0.001261 |
| iemocap4 | 0 | fixed0.95 | err_after | visual | 0.0347837 ± 0.00188108 |
| iemocap4 | 0 | reference | err_after | audio | 0.00171622 ± 7.70519e-05 |
| iemocap4 | 0 | reference | err_after | text | 0.00256227 ± 6.01979e-05 |
| iemocap4 | 0 | reference | err_after | visual | 0.00167965 ± 8.80841e-05 |
| iemocap4 | 0.1 | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.0 | err_decay | audio | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.0 | err_decay | text | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.0 | err_decay | visual | 1 ± 0 |
| iemocap4 | 0.1 | fixed0.2 | err_after | audio | 0.525303 ± 0.0280333 |
| iemocap4 | 0.1 | fixed0.2 | err_after | text | 0.722558 ± 0.013058 |
| iemocap4 | 0.1 | fixed0.2 | err_after | visual | 0.537433 ± 0.0161218 |
| iemocap4 | 0.1 | fixed0.2 | err_decay | audio | 0.543269 ± 0.0319471 |
| iemocap4 | 0.1 | fixed0.2 | err_decay | text | 0.726727 ± 0.0126963 |
| iemocap4 | 0.1 | fixed0.2 | err_decay | visual | 0.551438 ± 0.0223579 |
| iemocap4 | 0.1 | fixed0.4 | err_after | audio | 0.382114 ± 0.0219465 |
| iemocap4 | 0.1 | fixed0.4 | err_after | text | 0.552744 ± 0.0113825 |
| iemocap4 | 0.1 | fixed0.4 | err_after | visual | 0.378159 ± 0.013973 |
| iemocap4 | 0.1 | fixed0.4 | err_decay | audio | 0.399811 ± 0.0284148 |
| iemocap4 | 0.1 | fixed0.4 | err_decay | text | 0.560635 ± 0.0114329 |
| iemocap4 | 0.1 | fixed0.4 | err_decay | visual | 0.396042 ± 0.0192166 |
| iemocap4 | 0.1 | fixed0.6 | err_after | audio | 0.260346 ± 0.0151431 |
| iemocap4 | 0.1 | fixed0.6 | err_after | text | 0.384059 ± 0.00822997 |
| iemocap4 | 0.1 | fixed0.6 | err_after | visual | 0.251868 ± 0.0105202 |
| iemocap4 | 0.1 | fixed0.6 | err_decay | audio | 0.279544 ± 0.0216052 |
| iemocap4 | 0.1 | fixed0.6 | err_decay | text | 0.397955 ± 0.00933105 |
| iemocap4 | 0.1 | fixed0.6 | err_decay | visual | 0.274476 ± 0.0131292 |
| iemocap4 | 0.1 | fixed0.8 | err_after | audio | 0.137114 ± 0.00798862 |
| iemocap4 | 0.1 | fixed0.8 | err_after | text | 0.204035 ± 0.00440766 |
| iemocap4 | 0.1 | fixed0.8 | err_after | visual | 0.130874 ± 0.00590749 |
| iemocap4 | 0.1 | fixed0.8 | err_decay | audio | 0.162317 ± 0.012711 |
| iemocap4 | 0.1 | fixed0.8 | err_decay | text | 0.22881 ± 0.00658664 |
| iemocap4 | 0.1 | fixed0.8 | err_decay | visual | 0.162747 ± 0.00803307 |
| iemocap4 | 0.1 | fixed0.9 | err_after | audio | 0.0715459 ± 0.00416127 |
| iemocap4 | 0.1 | fixed0.9 | err_after | text | 0.106645 ± 0.00230066 |
| iemocap4 | 0.1 | fixed0.9 | err_after | visual | 0.0680282 ± 0.00315184 |
| iemocap4 | 0.1 | fixed0.9 | err_decay | audio | 0.102941 ± 0.00857481 |
| iemocap4 | 0.1 | fixed0.9 | err_decay | text | 0.139954 ± 0.00578354 |
| iemocap4 | 0.1 | fixed0.9 | err_decay | visual | 0.10752 ± 0.00638608 |
| iemocap4 | 0.1 | fixed0.95 | err_after | audio | 0.0372068 ± 0.00215937 |
| iemocap4 | 0.1 | fixed0.95 | err_after | text | 0.0554993 ± 0.00119334 |
| iemocap4 | 0.1 | fixed0.95 | err_after | visual | 0.0353398 ± 0.00165123 |
| iemocap4 | 0.1 | fixed0.95 | err_decay | audio | 0.0735643 ± 0.00730134 |
| iemocap4 | 0.1 | fixed0.95 | err_decay | text | 0.0947154 ± 0.00597258 |
| iemocap4 | 0.1 | fixed0.95 | err_decay | visual | 0.0803633 ± 0.0061165 |
| iemocap4 | 0.1 | reference | err_after | audio | 0.00170806 ± 7.90948e-05 |
| iemocap4 | 0.1 | reference | err_after | text | 0.00252873 ± 5.57751e-05 |
| iemocap4 | 0.1 | reference | err_after | visual | 0.00167039 ± 7.61694e-05 |
| iemocap4 | 0.1 | reference | err_decay | audio | 0.0486262 ± 0.00714248 |
| iemocap4 | 0.1 | reference | err_decay | text | 0.052608 ± 0.00691185 |
| iemocap4 | 0.1 | reference | err_decay | visual | 0.0571487 ± 0.00612599 |
| iemocap4 | 0.3 | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.0 | err_decay | audio | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.0 | err_decay | text | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.0 | err_decay | visual | 1 ± 0 |
| iemocap4 | 0.3 | fixed0.2 | err_after | audio | 0.546611 ± 0.0222034 |
| iemocap4 | 0.3 | fixed0.2 | err_after | text | 0.730331 ± 0.00981746 |
| iemocap4 | 0.3 | fixed0.2 | err_after | visual | 0.559426 ± 0.0182173 |
| iemocap4 | 0.3 | fixed0.2 | err_decay | audio | 0.56249 ± 0.024896 |
| iemocap4 | 0.3 | fixed0.2 | err_decay | text | 0.736183 ± 0.00994124 |
| iemocap4 | 0.3 | fixed0.2 | err_decay | visual | 0.577442 ± 0.023855 |
| iemocap4 | 0.3 | fixed0.4 | err_after | audio | 0.39588 ± 0.0182005 |
| iemocap4 | 0.3 | fixed0.4 | err_after | text | 0.557652 ± 0.00890163 |
| iemocap4 | 0.3 | fixed0.4 | err_after | visual | 0.393241 ± 0.0159564 |
| iemocap4 | 0.3 | fixed0.4 | err_decay | audio | 0.422138 ± 0.0227855 |
| iemocap4 | 0.3 | fixed0.4 | err_decay | text | 0.572443 ± 0.00981529 |
| iemocap4 | 0.3 | fixed0.4 | err_decay | visual | 0.421606 ± 0.0223021 |
| iemocap4 | 0.3 | fixed0.6 | err_after | audio | 0.26949 ± 0.0128557 |
| iemocap4 | 0.3 | fixed0.6 | err_after | text | 0.386657 ± 0.0067855 |
| iemocap4 | 0.3 | fixed0.6 | err_after | visual | 0.261631 ± 0.0119354 |
| iemocap4 | 0.3 | fixed0.6 | err_decay | audio | 0.30924 ± 0.0162989 |
| iemocap4 | 0.3 | fixed0.6 | err_decay | text | 0.41604 ± 0.00937858 |
| iemocap4 | 0.3 | fixed0.6 | err_decay | visual | 0.303258 ± 0.0170667 |
| iemocap4 | 0.3 | fixed0.8 | err_after | audio | 0.14197 ± 0.00687506 |
| iemocap4 | 0.3 | fixed0.8 | err_after | text | 0.204852 ± 0.00379648 |
| iemocap4 | 0.3 | fixed0.8 | err_after | visual | 0.13574 ± 0.00669664 |
| iemocap4 | 0.3 | fixed0.8 | err_decay | audio | 0.202759 ± 0.00999117 |
| iemocap4 | 0.3 | fixed0.8 | err_decay | text | 0.257807 ± 0.0102517 |
| iemocap4 | 0.3 | fixed0.8 | err_decay | visual | 0.198963 ± 0.0106277 |
| iemocap4 | 0.3 | fixed0.9 | err_after | audio | 0.0740802 ± 0.00359123 |
| iemocap4 | 0.3 | fixed0.9 | err_after | text | 0.106882 ± 0.00201127 |
| iemocap4 | 0.3 | fixed0.9 | err_after | visual | 0.0704831 ± 0.003573 |
| iemocap4 | 0.3 | fixed0.9 | err_decay | audio | 0.150897 ± 0.00872386 |
| iemocap4 | 0.3 | fixed0.9 | err_decay | text | 0.177599 ± 0.0121039 |
| iemocap4 | 0.3 | fixed0.9 | err_decay | visual | 0.150182 ± 0.00843066 |
| iemocap4 | 0.3 | fixed0.95 | err_after | audio | 0.0385143 ± 0.00186354 |
| iemocap4 | 0.3 | fixed0.95 | err_after | text | 0.05555 ± 0.00104944 |
| iemocap4 | 0.3 | fixed0.95 | err_after | visual | 0.0365854 ± 0.00187436 |
| iemocap4 | 0.3 | fixed0.95 | err_decay | audio | 0.126062 ± 0.0090326 |
| iemocap4 | 0.3 | fixed0.95 | err_decay | text | 0.13799 ± 0.0134795 |
| iemocap4 | 0.3 | fixed0.95 | err_decay | visual | 0.127217 ± 0.00820847 |
| iemocap4 | 0.3 | reference | err_after | audio | 0.00170951 ± 7.29734e-05 |
| iemocap4 | 0.3 | reference | err_after | text | 0.00245759 ± 4.27781e-05 |
| iemocap4 | 0.3 | reference | err_after | visual | 0.00166222 ± 8.91829e-05 |
| iemocap4 | 0.3 | reference | err_decay | audio | 0.105612 ± 0.00988667 |
| iemocap4 | 0.3 | reference | err_decay | text | 0.102267 ± 0.01508 |
| iemocap4 | 0.3 | reference | err_decay | visual | 0.108285 ± 0.00864944 |
| iemocap4 | 0.5 | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.0 | err_decay | audio | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.0 | err_decay | text | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.0 | err_decay | visual | 1 ± 0 |
| iemocap4 | 0.5 | fixed0.2 | err_after | audio | 0.570336 ± 0.0188473 |
| iemocap4 | 0.5 | fixed0.2 | err_after | text | 0.737544 ± 0.0117051 |
| iemocap4 | 0.5 | fixed0.2 | err_after | visual | 0.586682 ± 0.0156853 |
| iemocap4 | 0.5 | fixed0.2 | err_decay | audio | 0.584111 ± 0.0122548 |
| iemocap4 | 0.5 | fixed0.2 | err_decay | text | 0.748991 ± 0.0186872 |
| iemocap4 | 0.5 | fixed0.2 | err_decay | visual | 0.606225 ± 0.0170705 |
| iemocap4 | 0.5 | fixed0.4 | err_after | audio | 0.409591 ± 0.0163317 |
| iemocap4 | 0.5 | fixed0.4 | err_after | text | 0.562738 ± 0.0112096 |
| iemocap4 | 0.5 | fixed0.4 | err_after | visual | 0.411873 ± 0.0151153 |
| iemocap4 | 0.5 | fixed0.4 | err_decay | audio | 0.439594 ± 0.0110954 |
| iemocap4 | 0.5 | fixed0.4 | err_decay | text | 0.591447 ± 0.0170924 |
| iemocap4 | 0.5 | fixed0.4 | err_decay | visual | 0.446679 ± 0.015227 |
| iemocap4 | 0.5 | fixed0.6 | err_after | audio | 0.277111 ± 0.0121277 |
| iemocap4 | 0.5 | fixed0.6 | err_after | text | 0.389964 ± 0.00855451 |
| iemocap4 | 0.5 | fixed0.6 | err_after | visual | 0.273391 ± 0.0121265 |
| iemocap4 | 0.5 | fixed0.6 | err_decay | audio | 0.330653 ± 0.00940101 |
| iemocap4 | 0.5 | fixed0.6 | err_decay | text | 0.443471 ± 0.0120979 |
| iemocap4 | 0.5 | fixed0.6 | err_decay | visual | 0.330068 ± 0.0117173 |
| iemocap4 | 0.5 | fixed0.8 | err_after | audio | 0.145308 ± 0.00676185 |
| iemocap4 | 0.5 | fixed0.8 | err_after | text | 0.206305 ± 0.00474664 |
| iemocap4 | 0.5 | fixed0.8 | err_after | visual | 0.141372 ± 0.00711321 |
| iemocap4 | 0.5 | fixed0.8 | err_decay | audio | 0.234742 ± 0.0105841 |
| iemocap4 | 0.5 | fixed0.8 | err_decay | text | 0.297027 ± 0.007062 |
| iemocap4 | 0.5 | fixed0.8 | err_decay | visual | 0.232503 ± 0.00835743 |
| iemocap4 | 0.5 | fixed0.9 | err_after | audio | 0.0756793 ± 0.00358759 |
| iemocap4 | 0.5 | fixed0.9 | err_after | text | 0.107501 ± 0.00250565 |
| iemocap4 | 0.5 | fixed0.9 | err_after | visual | 0.073249 ± 0.00387714 |
| iemocap4 | 0.5 | fixed0.9 | err_decay | audio | 0.191059 ± 0.0123056 |
| iemocap4 | 0.5 | fixed0.9 | err_decay | text | 0.225135 ± 0.00558501 |
| iemocap4 | 0.5 | fixed0.9 | err_decay | visual | 0.189534 ± 0.0076245 |
| iemocap4 | 0.5 | fixed0.95 | err_after | audio | 0.0393038 ± 0.00187315 |
| iemocap4 | 0.5 | fixed0.95 | err_after | text | 0.0558089 ± 0.00130905 |
| iemocap4 | 0.5 | fixed0.95 | err_after | visual | 0.0379648 ± 0.00205607 |
| iemocap4 | 0.5 | fixed0.95 | err_decay | audio | 0.171159 ± 0.0133153 |
| iemocap4 | 0.5 | fixed0.95 | err_decay | text | 0.190684 ± 0.00542845 |
| iemocap4 | 0.5 | fixed0.95 | err_decay | visual | 0.170218 ± 0.00764908 |
| iemocap4 | 0.5 | reference | err_after | audio | 0.00168408 ± 6.90055e-05 |
| iemocap4 | 0.5 | reference | err_after | text | 0.00239179 ± 6.46826e-05 |
| iemocap4 | 0.5 | reference | err_after | visual | 0.00164696 ± 0.000101443 |
| iemocap4 | 0.5 | reference | err_decay | audio | 0.155689 ± 0.014281 |
| iemocap4 | 0.5 | reference | err_decay | text | 0.160484 ± 0.0057241 |
| iemocap4 | 0.5 | reference | err_decay | visual | 0.154966 ± 0.00802124 |
| iemocap4 | 0.7 | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.0 | err_decay | audio | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.0 | err_decay | text | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.0 | err_decay | visual | 1 ± 0 |
| iemocap4 | 0.7 | fixed0.2 | err_after | audio | 0.595079 ± 0.0217652 |
| iemocap4 | 0.7 | fixed0.2 | err_after | text | 0.7472 ± 0.00943596 |
| iemocap4 | 0.7 | fixed0.2 | err_after | visual | 0.607442 ± 0.0136752 |
| iemocap4 | 0.7 | fixed0.2 | err_decay | audio | 0.616795 ± 0.0183191 |
| iemocap4 | 0.7 | fixed0.2 | err_decay | text | 0.763671 ± 0.0118056 |
| iemocap4 | 0.7 | fixed0.2 | err_decay | visual | 0.635271 ± 0.0126916 |
| iemocap4 | 0.7 | fixed0.4 | err_after | audio | 0.424343 ± 0.0188341 |
| iemocap4 | 0.7 | fixed0.4 | err_after | text | 0.568379 ± 0.00834695 |
| iemocap4 | 0.7 | fixed0.4 | err_after | visual | 0.424149 ± 0.0125573 |
| iemocap4 | 0.7 | fixed0.4 | err_decay | audio | 0.46984 ± 0.0131621 |
| iemocap4 | 0.7 | fixed0.4 | err_decay | text | 0.605651 ± 0.0103371 |
| iemocap4 | 0.7 | fixed0.4 | err_decay | visual | 0.475085 ± 0.0144051 |
| iemocap4 | 0.7 | fixed0.6 | err_after | audio | 0.285966 ± 0.0128551 |
| iemocap4 | 0.7 | fixed0.6 | err_after | text | 0.39266 ± 0.00597989 |
| iemocap4 | 0.7 | fixed0.6 | err_after | visual | 0.279664 ± 0.0093565 |
| iemocap4 | 0.7 | fixed0.6 | err_decay | audio | 0.364447 ± 0.00550021 |
| iemocap4 | 0.7 | fixed0.6 | err_decay | text | 0.461022 ± 0.00759712 |
| iemocap4 | 0.7 | fixed0.6 | err_decay | visual | 0.36182 ± 0.01481 |
| iemocap4 | 0.7 | fixed0.8 | err_after | audio | 0.149604 ± 0.00642372 |
| iemocap4 | 0.7 | fixed0.8 | err_after | text | 0.20708 ± 0.00314374 |
| iemocap4 | 0.7 | fixed0.8 | err_after | visual | 0.143677 ± 0.0051624 |
| iemocap4 | 0.7 | fixed0.8 | err_decay | audio | 0.276994 ± 0.00581633 |
| iemocap4 | 0.7 | fixed0.8 | err_decay | text | 0.322861 ± 0.0069114 |
| iemocap4 | 0.7 | fixed0.8 | err_decay | visual | 0.272711 ± 0.0151755 |
| iemocap4 | 0.7 | fixed0.9 | err_after | audio | 0.0778311 ± 0.00323483 |
| iemocap4 | 0.7 | fixed0.9 | err_after | text | 0.107703 ± 0.00162679 |
| iemocap4 | 0.7 | fixed0.9 | err_after | visual | 0.074216 ± 0.00274151 |
| iemocap4 | 0.7 | fixed0.9 | err_decay | audio | 0.239548 ± 0.00904669 |
| iemocap4 | 0.7 | fixed0.9 | err_decay | text | 0.257514 ± 0.00773415 |
| iemocap4 | 0.7 | fixed0.9 | err_decay | visual | 0.236306 ± 0.0156593 |
| iemocap4 | 0.7 | fixed0.95 | err_after | audio | 0.0403891 ± 0.00164935 |
| iemocap4 | 0.7 | fixed0.95 | err_after | text | 0.0558366 ± 0.000845232 |
| iemocap4 | 0.7 | fixed0.95 | err_after | visual | 0.0384017 ± 0.00143627 |
| iemocap4 | 0.7 | fixed0.95 | err_decay | audio | 0.223319 ± 0.0106668 |
| iemocap4 | 0.7 | fixed0.95 | err_decay | text | 0.227063 ± 0.00831625 |
| iemocap4 | 0.7 | fixed0.95 | err_decay | visual | 0.220918 ± 0.0160016 |
| iemocap4 | 0.7 | reference | err_after | audio | 0.0016693 ± 6.27138e-05 |
| iemocap4 | 0.7 | reference | err_after | text | 0.00231336 ± 4.28198e-05 |
| iemocap4 | 0.7 | reference | err_after | visual | 0.00159837 ± 6.43679e-05 |
| iemocap4 | 0.7 | reference | err_decay | audio | 0.211337 ± 0.012172 |
| iemocap4 | 0.7 | reference | err_decay | text | 0.200956 ± 0.00888991 |
| iemocap4 | 0.7 | reference | err_decay | visual | 0.209545 ± 0.0164014 |
| iemocap4 | all | fixed0.0 | err_after | audio | 1 ± 0 |
| iemocap4 | all | fixed0.0 | err_after | text | 1 ± 0 |
| iemocap4 | all | fixed0.0 | err_after | visual | 1 ± 0 |
| iemocap4 | all | fixed0.2 | err_after | audio | 0.551178 ± 0.0232774 |
| iemocap4 | all | fixed0.2 | err_after | text | 0.731455 ± 0.011106 |
| iemocap4 | all | fixed0.2 | err_after | visual | 0.563884 ± 0.015961 |
| iemocap4 | all | fixed0.4 | err_after | audio | 0.397958 ± 0.0191562 |
| iemocap4 | all | fixed0.4 | err_after | text | 0.558475 ± 0.0100205 |
| iemocap4 | all | fixed0.4 | err_after | visual | 0.395866 ± 0.0142818 |
| iemocap4 | all | fixed0.6 | err_after | audio | 0.270073 ± 0.0134679 |
| iemocap4 | all | fixed0.6 | err_after | text | 0.387276 ± 0.00745346 |
| iemocap4 | all | fixed0.6 | err_after | visual | 0.262853 ± 0.0108946 |
| iemocap4 | all | fixed0.8 | err_after | audio | 0.141908 ± 0.00714626 |
| iemocap4 | all | fixed0.8 | err_after | text | 0.205196 ± 0.0040676 |
| iemocap4 | all | fixed0.8 | err_after | visual | 0.136081 ± 0.00617867 |
| iemocap4 | all | fixed0.9 | err_after | audio | 0.0739694 ± 0.00372042 |
| iemocap4 | all | fixed0.9 | err_after | text | 0.107056 ± 0.00213586 |
| iemocap4 | all | fixed0.9 | err_after | visual | 0.0705829 ± 0.00331343 |
| iemocap4 | all | fixed0.95 | err_after | audio | 0.0384372 ± 0.00192829 |
| iemocap4 | all | fixed0.95 | err_after | text | 0.055635 ± 0.00111189 |
| iemocap4 | all | fixed0.95 | err_after | visual | 0.0366151 ± 0.00174201 |
| iemocap4 | all | reference | err_after | audio | 0.00169743 ± 7.01652e-05 |
| iemocap4 | all | reference | err_after | text | 0.00245075 ± 4.98899e-05 |
| iemocap4 | all | reference | err_after | visual | 0.00165152 ± 8.19501e-05 |
| mosi | 0 | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | 0 | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | 0 | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | 0 | fixed0.2 | err_after | audio | 0.244764 ± 0.00748137 |
| mosi | 0 | fixed0.2 | err_after | text | 0.58373 ± 0.027516 |
| mosi | 0 | fixed0.2 | err_after | visual | 0.318306 ± 0.0220586 |
| mosi | 0 | fixed0.4 | err_after | audio | 0.11929 ± 0.00902978 |
| mosi | 0 | fixed0.4 | err_after | text | 0.422054 ± 0.0242193 |
| mosi | 0 | fixed0.4 | err_after | visual | 0.191876 ± 0.0208212 |
| mosi | 0 | fixed0.6 | err_after | audio | 0.0665504 ± 0.00744086 |
| mosi | 0 | fixed0.6 | err_after | text | 0.283315 ± 0.0176422 |
| mosi | 0 | fixed0.6 | err_after | visual | 0.119823 ± 0.0153108 |
| mosi | 0 | fixed0.8 | err_after | audio | 0.0312119 ± 0.00434002 |
| mosi | 0 | fixed0.8 | err_after | text | 0.146366 ± 0.00959239 |
| mosi | 0 | fixed0.8 | err_after | visual | 0.0597566 ± 0.00833081 |
| mosi | 0 | fixed0.9 | err_after | audio | 0.0156533 ± 0.00234321 |
| mosi | 0 | fixed0.9 | err_after | text | 0.075697 ± 0.00507376 |
| mosi | 0 | fixed0.9 | err_after | visual | 0.0305893 ± 0.00439198 |
| mosi | 0 | fixed0.95 | err_after | audio | 0.00803105 ± 0.00122732 |
| mosi | 0 | fixed0.95 | err_after | text | 0.0392791 ± 0.00263876 |
| mosi | 0 | fixed0.95 | err_after | visual | 0.0158166 ± 0.00228891 |
| mosi | 0 | reference | err_after | audio | 0.000629299 ± 0.000169865 |
| mosi | 0 | reference | err_after | text | 0.0018528 ± 9.22397e-05 |
| mosi | 0 | reference | err_after | visual | 0.00102834 ± 3.61136e-05 |
| mosi | 0.1 | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | 0.1 | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | 0.1 | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | 0.1 | fixed0.0 | err_decay | audio | 1 ± 0 |
| mosi | 0.1 | fixed0.0 | err_decay | text | 1 ± 0 |
| mosi | 0.1 | fixed0.0 | err_decay | visual | 1 ± 0 |
| mosi | 0.1 | fixed0.2 | err_after | audio | 0.267317 ± 0.00576024 |
| mosi | 0.1 | fixed0.2 | err_after | text | 0.591199 ± 0.0274328 |
| mosi | 0.1 | fixed0.2 | err_after | visual | 0.3343 ± 0.0200356 |
| mosi | 0.1 | fixed0.2 | err_decay | audio | 0.299049 ± 0.0151867 |
| mosi | 0.1 | fixed0.2 | err_decay | text | 0.597805 ± 0.0453456 |
| mosi | 0.1 | fixed0.2 | err_decay | visual | 0.341477 ± 0.022978 |
| mosi | 0.1 | fixed0.4 | err_after | audio | 0.131577 ± 0.00776759 |
| mosi | 0.1 | fixed0.4 | err_after | text | 0.426808 ± 0.0235252 |
| mosi | 0.1 | fixed0.4 | err_after | visual | 0.201156 ± 0.0192673 |
| mosi | 0.1 | fixed0.4 | err_decay | audio | 0.157882 ± 0.0208762 |
| mosi | 0.1 | fixed0.4 | err_decay | text | 0.436376 ± 0.0436361 |
| mosi | 0.1 | fixed0.4 | err_decay | visual | 0.208356 ± 0.0212584 |
| mosi | 0.1 | fixed0.6 | err_after | audio | 0.0740679 ± 0.00663571 |
| mosi | 0.1 | fixed0.6 | err_after | text | 0.286483 ± 0.0164425 |
| mosi | 0.1 | fixed0.6 | err_after | visual | 0.125948 ± 0.0141727 |
| mosi | 0.1 | fixed0.6 | err_decay | audio | 0.0998665 ± 0.0172814 |
| mosi | 0.1 | fixed0.6 | err_decay | text | 0.295912 ± 0.0299574 |
| mosi | 0.1 | fixed0.6 | err_decay | visual | 0.136009 ± 0.0144211 |
| mosi | 0.1 | fixed0.8 | err_after | audio | 0.0351606 ± 0.00392026 |
| mosi | 0.1 | fixed0.8 | err_after | text | 0.148027 ± 0.0086608 |
| mosi | 0.1 | fixed0.8 | err_after | visual | 0.0630823 ± 0.00762998 |
| mosi | 0.1 | fixed0.8 | err_decay | audio | 0.0656893 ± 0.0115498 |
| mosi | 0.1 | fixed0.8 | err_decay | text | 0.160109 ± 0.0145742 |
| mosi | 0.1 | fixed0.8 | err_decay | visual | 0.0787244 ± 0.00726042 |
| mosi | 0.1 | fixed0.9 | err_after | audio | 0.0177556 ± 0.0021282 |
| mosi | 0.1 | fixed0.9 | err_after | text | 0.0765277 ± 0.00455199 |
| mosi | 0.1 | fixed0.9 | err_after | visual | 0.0323653 ± 0.00398364 |
| mosi | 0.1 | fixed0.9 | err_decay | audio | 0.0531736 ± 0.00883303 |
| mosi | 0.1 | fixed0.9 | err_decay | text | 0.0930008 ± 0.00696862 |
| mosi | 0.1 | fixed0.9 | err_decay | visual | 0.0528809 ± 0.00468183 |
| mosi | 0.1 | fixed0.95 | err_after | audio | 0.00913679 ± 0.00112218 |
| mosi | 0.1 | fixed0.95 | err_after | text | 0.0396839 ± 0.00236409 |
| mosi | 0.1 | fixed0.95 | err_after | visual | 0.0167474 ± 0.00206233 |
| mosi | 0.1 | fixed0.95 | err_decay | audio | 0.0482116 ± 0.00773895 |
| mosi | 0.1 | fixed0.95 | err_decay | text | 0.06014 ± 0.00364386 |
| mosi | 0.1 | fixed0.95 | err_decay | visual | 0.0411881 ± 0.00413415 |
| mosi | 0.1 | reference | err_after | audio | 0.00062629 ± 0.000151972 |
| mosi | 0.1 | reference | err_after | text | 0.0018338 ± 7.39818e-05 |
| mosi | 0.1 | reference | err_after | visual | 0.00101713 ± 3.56224e-05 |
| mosi | 0.1 | reference | err_decay | audio | 0.0448148 ± 0.00699399 |
| mosi | 0.1 | reference | err_decay | text | 0.0306286 ± 0.00281972 |
| mosi | 0.1 | reference | err_decay | visual | 0.0323899 ± 0.00444536 |
| mosi | 0.3 | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | 0.3 | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | 0.3 | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | 0.3 | fixed0.0 | err_decay | audio | 1 ± 0 |
| mosi | 0.3 | fixed0.0 | err_decay | text | 1 ± 0 |
| mosi | 0.3 | fixed0.0 | err_decay | visual | 1 ± 0 |
| mosi | 0.3 | fixed0.2 | err_after | audio | 0.314115 ± 0.00766829 |
| mosi | 0.3 | fixed0.2 | err_after | text | 0.613871 ± 0.0231736 |
| mosi | 0.3 | fixed0.2 | err_after | visual | 0.374517 ± 0.017994 |
| mosi | 0.3 | fixed0.2 | err_decay | audio | 0.346493 ± 0.0268808 |
| mosi | 0.3 | fixed0.2 | err_decay | text | 0.628854 ± 0.0211261 |
| mosi | 0.3 | fixed0.2 | err_decay | visual | 0.398083 ± 0.0156402 |
| mosi | 0.3 | fixed0.4 | err_after | audio | 0.157199 ± 0.00768982 |
| mosi | 0.3 | fixed0.4 | err_after | text | 0.441552 ± 0.0202886 |
| mosi | 0.3 | fixed0.4 | err_after | visual | 0.22326 ± 0.0179386 |
| mosi | 0.3 | fixed0.4 | err_decay | audio | 0.189606 ± 0.0220416 |
| mosi | 0.3 | fixed0.4 | err_decay | text | 0.459418 ± 0.017213 |
| mosi | 0.3 | fixed0.4 | err_decay | visual | 0.24867 ± 0.0146524 |
| mosi | 0.3 | fixed0.6 | err_after | audio | 0.0884443 ± 0.0068524 |
| mosi | 0.3 | fixed0.6 | err_after | text | 0.295885 ± 0.0149575 |
| mosi | 0.3 | fixed0.6 | err_after | visual | 0.139531 ± 0.0133271 |
| mosi | 0.3 | fixed0.6 | err_decay | audio | 0.12668 ± 0.0204737 |
| mosi | 0.3 | fixed0.6 | err_decay | text | 0.318709 ± 0.0124726 |
| mosi | 0.3 | fixed0.6 | err_decay | visual | 0.171988 ± 0.00896656 |
| mosi | 0.3 | fixed0.8 | err_after | audio | 0.0419935 ± 0.00420792 |
| mosi | 0.3 | fixed0.8 | err_after | text | 0.152683 ± 0.00844952 |
| mosi | 0.3 | fixed0.8 | err_after | visual | 0.0703867 ± 0.00714149 |
| mosi | 0.3 | fixed0.8 | err_decay | audio | 0.092568 ± 0.0224542 |
| mosi | 0.3 | fixed0.8 | err_decay | text | 0.18321 ± 0.00894923 |
| mosi | 0.3 | fixed0.8 | err_decay | visual | 0.116227 ± 0.00603946 |
| mosi | 0.3 | fixed0.9 | err_after | audio | 0.0212251 ± 0.00232973 |
| mosi | 0.3 | fixed0.9 | err_after | text | 0.0788285 ± 0.00454765 |
| mosi | 0.3 | fixed0.9 | err_after | visual | 0.0362839 ± 0.00372051 |
| mosi | 0.3 | fixed0.9 | err_decay | audio | 0.081178 ± 0.0240665 |
| mosi | 0.3 | fixed0.9 | err_decay | text | 0.115938 ± 0.00848293 |
| mosi | 0.3 | fixed0.9 | err_decay | visual | 0.092509 ± 0.00893579 |
| mosi | 0.3 | fixed0.95 | err_after | audio | 0.0109215 ± 0.0012516 |
| mosi | 0.3 | fixed0.95 | err_after | text | 0.0408145 ± 0.00238463 |
| mosi | 0.3 | fixed0.95 | err_after | visual | 0.0188074 ± 0.00192435 |
| mosi | 0.3 | fixed0.95 | err_decay | audio | 0.0769548 ± 0.0249109 |
| mosi | 0.3 | fixed0.95 | err_decay | text | 0.08307 ± 0.00948129 |
| mosi | 0.3 | fixed0.95 | err_decay | visual | 0.0821954 ± 0.0108534 |
| mosi | 0.3 | reference | err_after | audio | 0.000631217 ± 0.000135631 |
| mosi | 0.3 | reference | err_after | text | 0.00182061 ± 6.8741e-05 |
| mosi | 0.3 | reference | err_after | visual | 0.00101862 ± 3.37141e-05 |
| mosi | 0.3 | reference | err_decay | audio | 0.0744068 ± 0.0258301 |
| mosi | 0.3 | reference | err_decay | text | 0.0535061 ± 0.0115501 |
| mosi | 0.3 | reference | err_decay | visual | 0.0748923 ± 0.0126406 |
| mosi | 0.5 | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | 0.5 | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | 0.5 | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | 0.5 | fixed0.0 | err_decay | audio | 1 ± 0 |
| mosi | 0.5 | fixed0.0 | err_decay | text | 1 ± 0 |
| mosi | 0.5 | fixed0.0 | err_decay | visual | 1 ± 0 |
| mosi | 0.5 | fixed0.2 | err_after | audio | 0.382149 ± 0.0119554 |
| mosi | 0.5 | fixed0.2 | err_after | text | 0.628709 ± 0.0259313 |
| mosi | 0.5 | fixed0.2 | err_after | visual | 0.429642 ± 0.0111085 |
| mosi | 0.5 | fixed0.2 | err_decay | audio | 0.412873 ± 0.0175765 |
| mosi | 0.5 | fixed0.2 | err_decay | text | 0.643295 ± 0.0298096 |
| mosi | 0.5 | fixed0.2 | err_decay | visual | 0.471202 ± 0.00592049 |
| mosi | 0.5 | fixed0.4 | err_after | audio | 0.19533 ± 0.0121707 |
| mosi | 0.5 | fixed0.4 | err_after | text | 0.446949 ± 0.0235968 |
| mosi | 0.5 | fixed0.4 | err_after | visual | 0.255392 ± 0.0131023 |
| mosi | 0.5 | fixed0.4 | err_decay | audio | 0.234276 ± 0.0173498 |
| mosi | 0.5 | fixed0.4 | err_decay | text | 0.472475 ± 0.0291468 |
| mosi | 0.5 | fixed0.4 | err_decay | visual | 0.300549 ± 0.00705724 |
| mosi | 0.5 | fixed0.6 | err_after | audio | 0.108038 ± 0.00925079 |
| mosi | 0.5 | fixed0.6 | err_after | text | 0.298089 ± 0.0172268 |
| mosi | 0.5 | fixed0.6 | err_after | visual | 0.157361 ± 0.0109585 |
| mosi | 0.5 | fixed0.6 | err_decay | audio | 0.157984 ± 0.0182932 |
| mosi | 0.5 | fixed0.6 | err_decay | text | 0.33559 ± 0.0237463 |
| mosi | 0.5 | fixed0.6 | err_decay | visual | 0.212492 ± 0.00645858 |
| mosi | 0.5 | fixed0.8 | err_after | audio | 0.0504829 ± 0.00526691 |
| mosi | 0.5 | fixed0.8 | err_after | text | 0.153666 ± 0.00925716 |
| mosi | 0.5 | fixed0.8 | err_after | visual | 0.0786476 ± 0.00622198 |
| mosi | 0.5 | fixed0.8 | err_decay | audio | 0.117475 ± 0.021099 |
| mosi | 0.5 | fixed0.8 | err_decay | text | 0.205785 ± 0.019284 |
| mosi | 0.5 | fixed0.8 | err_decay | visual | 0.154482 ± 0.00800418 |
| mosi | 0.5 | fixed0.9 | err_after | audio | 0.0253913 ± 0.00286941 |
| mosi | 0.5 | fixed0.9 | err_after | text | 0.0793197 ± 0.00488229 |
| mosi | 0.5 | fixed0.9 | err_after | visual | 0.0404112 ± 0.00327284 |
| mosi | 0.5 | fixed0.9 | err_decay | audio | 0.104375 ± 0.0227762 |
| mosi | 0.5 | fixed0.9 | err_decay | text | 0.141561 ± 0.0205757 |
| mosi | 0.5 | fixed0.9 | err_decay | visual | 0.131829 ± 0.01136 |
| mosi | 0.5 | fixed0.95 | err_after | audio | 0.013034 ± 0.00152722 |
| mosi | 0.5 | fixed0.95 | err_after | text | 0.0410333 ± 0.00255173 |
| mosi | 0.5 | fixed0.95 | err_after | visual | 0.020904 ± 0.00169451 |
| mosi | 0.5 | fixed0.95 | err_decay | audio | 0.0995296 ± 0.0236786 |
| mosi | 0.5 | fixed0.95 | err_decay | text | 0.11025 ± 0.0226846 |
| mosi | 0.5 | fixed0.95 | err_decay | visual | 0.122401 ± 0.0133685 |
| mosi | 0.5 | reference | err_after | audio | 0.000633273 ± 8.18248e-05 |
| mosi | 0.5 | reference | err_after | text | 0.00174633 ± 6.84633e-05 |
| mosi | 0.5 | reference | err_after | visual | 0.000994669 ± 4.74238e-05 |
| mosi | 0.5 | reference | err_decay | audio | 0.0964873 ± 0.0247301 |
| mosi | 0.5 | reference | err_decay | text | 0.0819438 ± 0.0255119 |
| mosi | 0.5 | reference | err_decay | visual | 0.116016 ± 0.0151836 |
| mosi | 0.7 | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | 0.7 | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | 0.7 | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | 0.7 | fixed0.0 | err_decay | audio | 1 ± 0 |
| mosi | 0.7 | fixed0.0 | err_decay | text | 1 ± 0 |
| mosi | 0.7 | fixed0.0 | err_decay | visual | 1 ± 0 |
| mosi | 0.7 | fixed0.2 | err_after | audio | 0.438334 ± 0.0176041 |
| mosi | 0.7 | fixed0.2 | err_after | text | 0.654667 ± 0.0215143 |
| mosi | 0.7 | fixed0.2 | err_after | visual | 0.471611 ± 0.0185509 |
| mosi | 0.7 | fixed0.2 | err_decay | audio | 0.488753 ± 0.01914 |
| mosi | 0.7 | fixed0.2 | err_decay | text | 0.679483 ± 0.0288433 |
| mosi | 0.7 | fixed0.2 | err_decay | visual | 0.517881 ± 0.0175176 |
| mosi | 0.7 | fixed0.4 | err_after | audio | 0.233165 ± 0.0161109 |
| mosi | 0.7 | fixed0.4 | err_after | text | 0.462876 ± 0.0173494 |
| mosi | 0.7 | fixed0.4 | err_after | visual | 0.282539 ± 0.016429 |
| mosi | 0.7 | fixed0.4 | err_decay | audio | 0.296298 ± 0.0223603 |
| mosi | 0.7 | fixed0.4 | err_decay | text | 0.503696 ± 0.0244342 |
| mosi | 0.7 | fixed0.4 | err_decay | visual | 0.343345 ± 0.0142571 |
| mosi | 0.7 | fixed0.6 | err_after | audio | 0.128468 ± 0.0123022 |
| mosi | 0.7 | fixed0.6 | err_after | text | 0.305254 ± 0.0122152 |
| mosi | 0.7 | fixed0.6 | err_after | visual | 0.172524 ± 0.0120448 |
| mosi | 0.7 | fixed0.6 | err_decay | audio | 0.203831 ± 0.0248939 |
| mosi | 0.7 | fixed0.6 | err_decay | text | 0.361783 ± 0.0179393 |
| mosi | 0.7 | fixed0.6 | err_decay | visual | 0.249929 ± 0.0105084 |
| mosi | 0.7 | fixed0.8 | err_after | audio | 0.0589836 ± 0.00710518 |
| mosi | 0.7 | fixed0.8 | err_after | text | 0.155691 ± 0.0067439 |
| mosi | 0.7 | fixed0.8 | err_after | visual | 0.0853319 ± 0.00657411 |
| mosi | 0.7 | fixed0.8 | err_decay | audio | 0.153539 ± 0.0274496 |
| mosi | 0.7 | fixed0.8 | err_decay | text | 0.232876 ± 0.0159129 |
| mosi | 0.7 | fixed0.8 | err_decay | visual | 0.189409 ± 0.0113039 |
| mosi | 0.7 | fixed0.9 | err_after | audio | 0.0293628 ± 0.00386997 |
| mosi | 0.7 | fixed0.9 | err_after | text | 0.0799403 ± 0.00361376 |
| mosi | 0.7 | fixed0.9 | err_after | visual | 0.0436352 ± 0.00345295 |
| mosi | 0.7 | fixed0.9 | err_decay | audio | 0.137806 ± 0.0288272 |
| mosi | 0.7 | fixed0.9 | err_decay | text | 0.17165 ± 0.0183252 |
| mosi | 0.7 | fixed0.9 | err_decay | visual | 0.166911 ± 0.0135609 |
| mosi | 0.7 | fixed0.95 | err_after | audio | 0.0149883 ± 0.00205669 |
| mosi | 0.7 | fixed0.95 | err_after | text | 0.0412252 ± 0.00190419 |
| mosi | 0.7 | fixed0.95 | err_after | visual | 0.022508 ± 0.00179756 |
| mosi | 0.7 | fixed0.95 | err_decay | audio | 0.132092 ± 0.0296245 |
| mosi | 0.7 | fixed0.95 | err_decay | text | 0.142413 ± 0.0203758 |
| mosi | 0.7 | fixed0.95 | err_decay | visual | 0.157906 ± 0.014788 |
| mosi | 0.7 | reference | err_after | audio | 0.000643122 ± 8.72457e-05 |
| mosi | 0.7 | reference | err_after | text | 0.0016845 ± 6.64864e-05 |
| mosi | 0.7 | reference | err_after | visual | 0.000967532 ± 7.31507e-05 |
| mosi | 0.7 | reference | err_decay | audio | 0.128364 ± 0.0306311 |
| mosi | 0.7 | reference | err_decay | text | 0.116163 ± 0.022738 |
| mosi | 0.7 | reference | err_decay | visual | 0.15197 ± 0.0158108 |
| mosi | all | fixed0.0 | err_after | audio | 1 ± 0 |
| mosi | all | fixed0.0 | err_after | text | 1 ± 0 |
| mosi | all | fixed0.0 | err_after | visual | 1 ± 0 |
| mosi | all | fixed0.2 | err_after | audio | 0.329336 ± 0.00795952 |
| mosi | all | fixed0.2 | err_after | text | 0.614435 ± 0.0237436 |
| mosi | all | fixed0.2 | err_after | visual | 0.385675 ± 0.0166894 |
| mosi | all | fixed0.4 | err_after | audio | 0.167312 ± 0.00947619 |
| mosi | all | fixed0.4 | err_after | text | 0.440048 ± 0.0206239 |
| mosi | all | fixed0.4 | err_after | visual | 0.230845 ± 0.0171226 |
| mosi | all | fixed0.6 | err_after | audio | 0.0931138 ± 0.00784514 |
| mosi | all | fixed0.6 | err_after | text | 0.293805 ± 0.0149369 |
| mosi | all | fixed0.6 | err_after | visual | 0.143037 ± 0.0129992 |
| mosi | all | fixed0.8 | err_after | audio | 0.0435665 ± 0.00462722 |
| mosi | all | fixed0.8 | err_after | text | 0.151287 ± 0.00815427 |
| mosi | all | fixed0.8 | err_after | visual | 0.071441 ± 0.00710442 |
| mosi | all | fixed0.9 | err_after | audio | 0.0218776 ± 0.00252776 |
| mosi | all | fixed0.9 | err_after | text | 0.0780627 ± 0.00432901 |
| mosi | all | fixed0.9 | err_after | visual | 0.036657 ± 0.00372405 |
| mosi | all | fixed0.95 | err_after | audio | 0.0112223 ± 0.00134345 |
| mosi | all | fixed0.95 | err_after | text | 0.0404072 ± 0.0022583 |
| mosi | all | fixed0.95 | err_after | visual | 0.0189567 ± 0.00193187 |
| mosi | all | reference | err_after | audio | 0.00063264 ± 0.000120511 |
| mosi | all | reference | err_after | text | 0.00178761 ± 5.47305e-05 |
| mosi | all | reference | err_after | visual | 0.00100526 ± 3.96161e-05 |

All per-seed sampled best points, every adjacent difference, and matched seed differences at each sampled-mean peak versus its neighbors are in per_seed_best.csv, per_seed_adjacent.csv, and per_seed_peak_support.csv. CSV metrics use raw units. Existing checkpoint-selection bias remains; error diagnostics alone do not establish causality.
