# Coupled Source Role — partial completion

INTERNAL TEST-ORACLE SCREEN

Observed2026-10-20T13:37:29Z: MOSI completed150/150epochs; MOSEI running47/60,
without reported failure. Do not interpret the partial MOSEI score as final.

## Completed MOSI seed66

| Model |8-rate ACC|8-rate W-F1|High ACC|High W-F1|
|---|---:|---:|---:|---:|
|Original Flat|79.1159|79.1280|73.2724|73.3381|
|Coupled Source Role|79.1921|79.0506|73.1707|73.1139|
|Delta pp|+0.0762|−0.0774|−0.1016|−0.2242|

No overall W-F1 improvement in this seed66 screening. Small aggregate decline
does not establish statistical inferiority across seeds. Positive rates0.0,
0.1,0.4,0.5 are offset chiefly by0.3(−1.3923pp) and0.7(−1.3495pp).
No automatic extra seeds or parameter search is authorized.

All8selected prediction NPZs were independently checked against selection
hashes, exact original Flat cohorts/masks, earliest per-rate TEST W-F1 maximum,
and ACC/W-F1/MSE recomputation. Maximum metric discrepancy0. All150epochs
passed actual training-mask/update/LR pairing checks. All8BEST weights and
complete last recovery state remain remote. No new inference or training was
used for verification; no existing predictions or masks were overwritten.

Full per-rate scores/provenance: completed/CMUMOSI/{RESULT.md,METRICS.csv,
SUMMARY.json,FINAL_VERIFICATION.json}. New parameter count96,487;
wall1662.781seconds,peakallocated3238.286MiB under shared-GPU conditions.

## MOSEI — still running at observation

Epoch47/60 current per-rate best:8-rate ACC83.3998,W-F183.2205;
high ACC80.3522,W-F179.9627. All47completedepochs passed the training
mask/update/LR pairing checks; all14new parameter tensors had nonzero gradients.
Completed Flat reference:8-rate ACC83.4927,W-F183.2791;high ACC80.5173,
W-F180.1817. Current W-F1 deltas:-.0586pp overall/-.2191pp high.
The Flat47epoch prefix already contains these same best scores. This is still
partial method evidence,not a final60epoch ranking or multi-seed conclusion.
This MOSEI run is the original source-role residual;no direct2*sigmoid MOSEI
or extra MOSEI seeds were launched. Per-rate progress:progress/CMUMOSEI.

Unchanged original GCNet random missing,scalarMSE,one-stage originalFlat1600
plus source-role calibration,sourcef6bc52e. No Router despite legacy metric key.
All numbers are exploratory per-rate Test-oracle, not a single deployment
checkpoint or independent-test generalization result.
