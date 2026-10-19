# TASR F2 Adapter LR screen

INTERNAL DIAGNOSTIC ONLY; validation-selected adapter; historical Test-oracle backbone

Completed 6/6. Only LR changes:3e-6/1e-5/3e-5; previous3e-4 retained, not rerun.

One fixed backbone per dataset; seed66, IEMOCAP6 fold5 only;30epochs.
Checkpoint selection remains validation random8 ACC, then CE. No test-based selection.

| Dataset | LR | Epoch | Random8 ACC | UA | W-F1 | Persistent6 ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MELD | 3e-06 | 30 | 61.3458 | 38.8671 | 58.6523 | 57.6054 | 32.0406 | 50.8506 |
| MELD | 1e-05 | 30 | 61.3554 | 38.8700 | 58.6599 | 57.6054 | 32.0406 | 50.8521 |
| MELD | 3e-05 | 19 | 61.3937 | 38.9123 | 58.6945 | 57.5990 | 32.0443 | 50.8603 |
| IEMOCAPSix | 3e-06 | 30 | 62.6155 | 62.0922 | 62.3830 | 49.7946 | 49.8991 | 48.9039 |
| IEMOCAPSix | 1e-05 | 30 | 62.6232 | 62.1067 | 62.3936 | 49.8049 | 49.9064 | 48.9144 |
| IEMOCAPSix | 3e-05 | 21 | 62.6078 | 62.0806 | 62.3796 | 49.7946 | 49.8901 | 48.9046 |

Per-rate/pattern, high-missing metrics, diagnostics and paired corrections/harms are in JSON/CSV.
Small single-seed LR exploration does not establish effectiveness; do not expand automatically.
