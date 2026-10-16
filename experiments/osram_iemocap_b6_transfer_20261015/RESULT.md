# Original-protocol IEMOCAP B6 transfer

INTERNAL PER-RATE TEST-ORACLE ACC; selected B6 settings transferred from MELD

Completed fold runs: 30/30. Status: complete.

Fold5 results are stored separately in FOLD5.json. Fivefold averages appear only after all five folds and three seeds finish.

| Dataset | Scope | ACC | UA | W-F1 |
|---|---|---:|---:|---:|
| IEMOCAPFour | fold5 | 82.830 ± 0.219 | 82.049 ± 0.492 | 82.733 ± 0.203 |
| IEMOCAPFour | fivefold | 79.287 ± 0.127 | 79.564 ± 0.090 | 79.160 ± 0.117 |
| IEMOCAPSix | fold5 | 64.944 ± 0.453 | 62.084 ± 1.220 | 64.236 ± 0.937 |
| IEMOCAPSix | fivefold | 61.734 ± 0.091 | 60.119 ± 0.287 | 61.350 ± 0.111 |

Values are equal-fold, then three-seed mean ± sample SD. Test-best uses separate ACC-selected epochs at eight missing rates.
Old exact-selection-epoch baseline UA is unavailable. Fresh final reevaluation and old baseline ACC are separate evidence.

## Verified original Flat comparison

Same100-epoch, fivefold/three-seed/eight-rate ACC-selected Test-best rule:

| Dataset | Original Flat ACC | B6 ACC | Difference(pp) |
|---|---:|---:|---:|
| IEMOCAP4 |79.261|79.287|+0.026|
| IEMOCAP6 |61.625|61.734|+0.109|

The remembered Flat77.757/60.495 values belong to the later fresh-mask retest,
not Test-best. They must not be compared with B6's selected scores as if the
difference were a method gain. No large ACC improvement is established.
Per-seed results, retest-to-retest comparison and exact source fields are in
BASELINE_COMPARISON.md. Original and B6 masks are not exact sample-paired.
