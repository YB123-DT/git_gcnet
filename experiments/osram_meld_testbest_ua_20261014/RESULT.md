# MELD corresponding Test-best UA — verified

INTERNAL DIAGNOSTIC ONLY. Seven retained model sources, eight rates each.
All 56 selected checkpoint evaluations reproduced their stored ACC and W-F1
within1e-10. All 56 final saved availability arrays were reconstructed exactly
from the historical NumPy draw stream. Therefore the UA below corresponds to
the same checkpoint AND masks that established each original ACC-best score.
UA is macro recall over the fixed seven emotion classes, not macro F1 and not
an independently selected best-UA score. All columns are percentages.

## Completed Flat width screen, seed66, 50 epochs

| Adapter width | ACC mean | UA mean | W-F1 mean | High ACC | High UA | High W-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1600 reference reproduction | 60.675 | 35.436 | 56.807 | 56.411 | 29.322 | 50.691 |
| 512 | 60.766 | 37.359 | 57.150 | 56.539 | 28.943 | 50.297 |
| 1024 | 60.623 | 35.402 | 56.380 | 56.577 | 29.281 | 50.383 |
| 2048 | 60.757 | 35.653 | 56.780 | 56.999 | 28.962 | 50.852 |

Width512 minus1600: +0.091 ACC, +1.923 UA, +0.343 W-F1; high-missing UA
is -0.379 and W-F1 is -0.394. This is a single-seed capacity screen, not
evidence that a write/read mechanism was improved. Only hidden width changed;
output1600, original GCNet masks, epochs50/batch32/lr1e-3 remained unchanged.

## Remove-Local-shortcut: actual selected-mask three-seed results

| Seed | ACC | Corresponding UA | W-F1 |
|---:|---:|---:|---:|
| 66 | 60.560 | 35.404 | 56.400 |
| 67 | 60.517 | 34.783 | 56.238 |
| 68 | 60.584 | 36.344 | 57.018 |
| Mean | 60.554 | 35.510 | 56.552 |

The historical original Flat three-seed ACC60.498 has no recoverable
corresponding UA without the deleted original weights. Its earlier comparison
using final-mask UA is withdrawn as a main-score comparison. The reproduced
Flat seed66 has ACC60.675/UA35.436; it is not the deleted historical seed66
with ACC60.637. Do not combine original scores with reproduction UA.

Original-code reproductions for Flat seeds67/68 are being prepared to complete
a fresh three-seed same-selection-mask comparison. Report their new scores
separately from the historical originals.

## Artifacts and verification

- Full checked values and checkpoint hashes: `SUMMARY.json`.
- Prediction/mask artifacts remain on biggpu at
  `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0` through `source_6`.
- Original sampler `gcnet-random-mask-v1` was replayed without alteration;
  source masks/weights were not overwritten. No optimizer or model training
  ran during UA recovery.
- The incorrect-for-main-report interim width file `PROGRESS_UA.json` was
  deleted from the current branch; Git history preserves it for audit.
- Mask-order regression and baseline restoration config test: 2 passed on
  biggpu. Actual selected ACC/W-F1 parity provides a separate real-data gate.
