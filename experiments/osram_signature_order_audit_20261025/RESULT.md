# Frozen Signature order / second-level area audit

MOSI seed66; eight original per-rate Test-best checkpoints; INTERNAL mechanism audit.
No training, new selection, masks, CHM updates, Local/Base/Gap changes or labels as inputs.
Controls change only the extra Signature descriptor of each fixed frame.

| Rate | Mode | N | ACC | WF1 | MSE | ΔMSE vs normal | Corrections / harms vs normal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | flat | 686 | 87.6524 | 87.6299 | 1.151984 | +0.115323 | 3 / 5 |
| 0.0 | normal | 686 | 87.9573 | 87.9544 | 1.036661 | +0.000000 | 0 / 0 |
| 0.0 | degree2_no_area | 686 | 87.6524 | 87.6494 | 1.030375 | -0.006286 | 1 / 3 |
| 0.0 | interior_shuffle_6601 | 686 | 87.8049 | 87.7988 | 1.035014 | -0.001647 | 0 / 1 |
| 0.0 | interior_shuffle_6602 | 686 | 87.8049 | 87.7988 | 1.035513 | -0.001148 | 0 / 1 |
| 0.0 | interior_shuffle_6603 | 686 | 87.9573 | 87.9544 | 1.034941 | -0.001720 | 0 / 0 |
| 0.1 | flat | 686 | 84.9085 | 84.7698 | 1.150175 | +0.009473 | 2 / 5 |
| 0.1 | normal | 686 | 85.3659 | 85.1891 | 1.140702 | +0.000000 | 0 / 0 |
| 0.1 | degree2_no_area | 686 | 85.3659 | 85.1891 | 1.140646 | -0.000056 | 0 / 0 |
| 0.1 | interior_shuffle_6601 | 686 | 85.3659 | 85.1891 | 1.140695 | -0.000007 | 0 / 0 |
| 0.1 | interior_shuffle_6602 | 686 | 85.3659 | 85.1891 | 1.140687 | -0.000015 | 0 / 0 |
| 0.1 | interior_shuffle_6603 | 686 | 85.3659 | 85.1891 | 1.140699 | -0.000003 | 0 / 0 |
| 0.2 | flat | 686 | 82.9268 | 82.9641 | 1.374416 | +0.031123 | 3 / 2 |
| 0.2 | normal | 686 | 82.7744 | 82.8152 | 1.343293 | +0.000000 | 0 / 0 |
| 0.2 | degree2_no_area | 686 | 82.9268 | 82.9641 | 1.342368 | -0.000925 | 1 / 0 |
| 0.2 | interior_shuffle_6601 | 686 | 82.7744 | 82.8152 | 1.343151 | -0.000142 | 0 / 0 |
| 0.2 | interior_shuffle_6602 | 686 | 82.7744 | 82.8152 | 1.343434 | +0.000141 | 0 / 0 |
| 0.2 | interior_shuffle_6603 | 686 | 82.7744 | 82.8152 | 1.343423 | +0.000130 | 0 / 0 |
| 0.3 | flat | 686 | 81.7073 | 81.7781 | 1.329429 | +0.006570 | 1 / 2 |
| 0.3 | normal | 686 | 81.8598 | 81.9217 | 1.322859 | +0.000000 | 0 / 0 |
| 0.3 | degree2_no_area | 686 | 81.8598 | 81.9217 | 1.321910 | -0.000949 | 0 / 0 |
| 0.3 | interior_shuffle_6601 | 686 | 81.8598 | 81.9217 | 1.322913 | +0.000054 | 0 / 0 |
| 0.3 | interior_shuffle_6602 | 686 | 81.8598 | 81.9217 | 1.323066 | +0.000207 | 0 / 0 |
| 0.3 | interior_shuffle_6603 | 686 | 81.8598 | 81.9217 | 1.322874 | +0.000015 | 0 / 0 |
| 0.4 | flat | 686 | 78.6585 | 78.4870 | 1.541472 | +0.042321 | 18 / 21 |
| 0.4 | normal | 686 | 79.1159 | 79.0143 | 1.499151 | +0.000000 | 0 / 0 |
| 0.4 | degree2_no_area | 686 | 78.0488 | 77.9199 | 1.451309 | -0.047842 | 10 / 17 |
| 0.4 | interior_shuffle_6601 | 686 | 79.4207 | 79.3595 | 1.485142 | -0.014009 | 10 / 8 |
| 0.4 | interior_shuffle_6602 | 686 | 79.7256 | 79.6772 | 1.486895 | -0.012256 | 11 / 7 |
| 0.4 | interior_shuffle_6603 | 686 | 79.1159 | 79.0279 | 1.496575 | -0.002576 | 9 / 9 |
| 0.5 | flat | 686 | 76.6768 | 76.8136 | 1.686353 | +0.074183 | 36 / 41 |
| 0.5 | normal | 686 | 77.4390 | 77.4278 | 1.612170 | +0.000000 | 0 / 0 |
| 0.5 | degree2_no_area | 686 | 76.8293 | 76.7805 | 1.591859 | -0.020311 | 5 / 9 |
| 0.5 | interior_shuffle_6601 | 686 | 77.2866 | 77.2323 | 1.613315 | +0.001145 | 3 / 4 |
| 0.5 | interior_shuffle_6602 | 686 | 77.2866 | 77.2323 | 1.611945 | -0.000226 | 3 / 4 |
| 0.5 | interior_shuffle_6603 | 686 | 76.9817 | 76.9397 | 1.614170 | +0.002000 | 3 / 6 |
| 0.6 | flat | 686 | 73.6280 | 73.6467 | 1.806606 | +0.000521 | 0 / 4 |
| 0.6 | normal | 686 | 74.2378 | 74.2560 | 1.806085 | +0.000000 | 0 / 0 |
| 0.6 | degree2_no_area | 686 | 74.2378 | 74.2560 | 1.806250 | +0.000165 | 0 / 0 |
| 0.6 | interior_shuffle_6601 | 686 | 74.2378 | 74.2560 | 1.806078 | -0.000007 | 0 / 0 |
| 0.6 | interior_shuffle_6602 | 686 | 74.2378 | 74.2560 | 1.806091 | +0.000006 | 0 / 0 |
| 0.6 | interior_shuffle_6603 | 686 | 74.2378 | 74.2560 | 1.806116 | +0.000031 | 0 / 0 |
| 0.7 | flat | 686 | 70.1220 | 69.5945 | 1.929045 | -0.022938 | 36 / 36 |
| 0.7 | normal | 686 | 70.1220 | 70.1984 | 1.951983 | +0.000000 | 0 / 0 |
| 0.7 | degree2_no_area | 686 | 69.9695 | 70.0291 | 1.951331 | -0.000652 | 2 / 3 |
| 0.7 | interior_shuffle_6601 | 686 | 70.1220 | 70.1984 | 1.953203 | +0.001220 | 1 / 1 |
| 0.7 | interior_shuffle_6602 | 686 | 70.1220 | 70.1984 | 1.952949 | +0.000966 | 1 / 1 |
| 0.7 | interior_shuffle_6603 | 686 | 70.1220 | 70.1984 | 1.952993 | +0.001010 | 1 / 1 |

## Eight-rate arithmetic means

| Mode | ACC | WF1 | MSE |
| --- | ---: | ---: | ---: |
| flat | 79.535061 | 79.460447 | 1.496185 |
| normal | 79.858994 | 79.847119 | 1.464113 |
| degree2_no_area | 79.611280 | 79.588720 | 1.454506 |
| interior_shuffle_6601 | 79.858994 | 79.846378 | 1.462439 |
| interior_shuffle_6602 | 79.897104 | 79.886088 | 1.462572 |
| interior_shuffle_6603 | 79.801829 | 79.787800 | 1.463974 |

## Interpretation limits

- CHM is already order-sensitive. These interventions test incremental path information, not an order-blind baseline.
- First-level signature under the fixed zero basepoint is an endpoint descriptor, not a temporal mean.
- Interior shuffle keeps observed tokens, counts, endpoints and fixed time coordinates, but changes their interior content/time correspondence.
- Removing antisymmetric second-level area preserves first level and symmetric outer-product terms. It can be out-of-distribution.
- Same-checkpoint interference is mechanism sensitivity, not proof of necessity or superiority to an equally trained order-free branch.
- Test-oracle and seed66 limitations remain; shuffled draws are not independent training seeds.
- Prediction bootstrap intervals condition on the trained checkpoint and test mask; they do not correct selection bias.
