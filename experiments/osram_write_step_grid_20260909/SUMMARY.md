# Frozen write-step grid

INTERNAL DIAGNOSTIC ONLY. Seeds 66–70; rates 0, 0.1, 0.3, 0.5, 0.7; eta 1, 0.95, 0.90, 0.80.

Weighted F1 is shown in percent; increments are percentage points. Mean ± sample SD uses five equally weighted seeds. `all` first averages the five rates within each seed. No rates/heads are treated as independent replicates.

IEMOCAP4 eta 1/0.90 are inherited in place. Checkpoint identity, config, epoch/selection metadata, weight-unchanged flags and same-seed/per-rate masks across all four modes passed metadata validation. This is not an independent checkpoint-file rehash. Exact source paths are in provenance.csv; no inherited raw artifacts were duplicated.

## iemocap4: weighted F1 (%)

| Rate | eta 1 | eta .95 | eta .90 | eta .80 |
|---|---|---|---|---|
| 0 | 84.334 ± 0.473 | 84.452 ± 0.452 | 84.471 ± 0.470 | 84.974 ± 0.359 |
| 0.1 | 83.924 ± 0.619 | 84.159 ± 0.527 | 84.209 ± 0.466 | 84.342 ± 0.344 |
| 0.3 | 81.993 ± 0.598 | 82.173 ± 0.557 | 82.193 ± 0.616 | 82.520 ± 0.672 |
| 0.5 | 80.118 ± 1.285 | 80.197 ± 1.242 | 80.339 ± 1.113 | 80.533 ± 1.108 |
| 0.7 | 77.750 ± 1.682 | 77.990 ± 1.645 | 78.122 ± 1.623 | 78.451 ± 1.510 |
| all | 81.624 ± 0.538 | 81.794 ± 0.534 | 81.867 ± 0.520 | 82.164 ± 0.514 |

Descriptive best grid point(s), five-rate mean: fixed0.8. This is not an optimality claim.

| Seed | eta 1 | eta .95 | eta .90 | eta .80 | 1→.95 | .95→.90 | .90→.80 |
|---|---|---|---|---|---|---|---|
| 66 | 81.784 | 81.959 | 82.011 | 82.521 | 0.175 | 0.051 | 0.510 |
| 67 | 81.447 | 81.544 | 81.546 | 81.751 | 0.097 | 0.002 | 0.205 |
| 68 | 81.647 | 81.771 | 81.834 | 81.975 | 0.124 | 0.064 | 0.141 |
| 69 | 80.879 | 81.126 | 81.288 | 81.698 | 0.247 | 0.163 | 0.410 |
| 70 | 82.363 | 82.572 | 82.655 | 82.874 | 0.208 | 0.083 | 0.220 |
## mosi: weighted F1 (%)

| Rate | eta 1 | eta .95 | eta .90 | eta .80 |
|---|---|---|---|---|
| 0 | 86.786 ± 0.610 | 86.786 ± 0.610 | 86.755 ± 0.547 | 86.812 ± 0.519 |
| 0.1 | 84.797 ± 1.154 | 84.827 ± 1.187 | 84.707 ± 1.140 | 84.646 ± 1.168 |
| 0.3 | 79.620 ± 0.998 | 79.714 ± 0.982 | 79.751 ± 1.075 | 80.056 ± 0.950 |
| 0.5 | 76.322 ± 1.645 | 76.286 ± 1.690 | 76.273 ± 1.681 | 76.208 ± 1.528 |
| 0.7 | 72.806 ± 3.232 | 73.071 ± 3.167 | 73.236 ± 2.936 | 73.280 ± 2.675 |
| all | 80.066 ± 0.541 | 80.137 ± 0.506 | 80.145 ± 0.365 | 80.201 ± 0.341 |

Descriptive best grid point(s), five-rate mean: fixed0.8. This is not an optimality claim.

| Seed | eta 1 | eta .95 | eta .90 | eta .80 | 1→.95 | .95→.90 | .90→.80 |
|---|---|---|---|---|---|---|---|
| 66 | 79.779 | 79.773 | 79.929 | 80.053 | -0.006 | 0.157 | 0.124 |
| 67 | 80.811 | 80.895 | 80.715 | 80.736 | 0.084 | -0.180 | 0.021 |
| 68 | 79.455 | 79.646 | 79.866 | 79.943 | 0.191 | 0.220 | 0.078 |
| 69 | 80.416 | 80.374 | 80.308 | 80.337 | -0.041 | -0.067 | 0.030 |
| 70 | 79.871 | 79.996 | 79.905 | 79.932 | 0.124 | -0.091 | 0.028 |

## Old-read and new-write errors

Old error is the actual-read `retention.err_decay` (NO_HISTORY excluded), not err_post. New error is `current_observed_write_fit.err_after`. Each evaluator run-level mean is one observation; five seeds are equally weighted and modalities remain separate. Undefined rate-zero retention is omitted, never zero-filled. No five-rate old-error aggregate is fabricated when rate zero is undefined.

| Dataset | Rate | Mode | Error | Modality | Mean ± SD |
|---|---|---|---|---|---|
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

## Frozen operational retrain gate

Rule: on BOTH datasets, eta .90 has greater five-rate mean Wf1 than eta 1 and positive paired gains in ≥3/5 seeds; eta .80 has lower mean Wf1 than eta .90 on BOTH datasets. Strict comparisons; no significance or extra effect-size threshold.

| Dataset | .90−1 (pp) | Positive seeds | .80−.90 (pp) | Pass |
|---|---|---|---|---|
| iemocap4 | 0.243 | 5/5 | 0.297 | False |
| mosi | 0.078 | 3/5 | 0.056 | False |

Gate passed: **False**. Report only: no training, checkpoint selection, or tuning performed.

Per-rate/per-seed points and adjacent increments are in per_seed_shape.csv; all task metrics and selected error diagnostics are in per_seed_metrics.csv and summary.csv (raw units). Five paired seeds provide descriptive evidence only. Existing checkpoint-selection bias remains; error diagnostics alone do not establish downstream causality.
