# MELD fixed small-LR three-seed confirmation

INTERNAL TEST-ORACLE SCREEN; ACC-selected corresponding UA/W-F1.

Completed: both added seeds finished50epochs without failures. All seeds use
fixed LR1e-5 / Adam weight_decay2e-4, original Flat1600 + Local Skip, the accepted
ECERC features, cmerc-repro and GCNet-random-mask-v1. Seed66 is reused, not retrained.
Each seed retains8 BEST weights, linked selected predictions and full last state.

Eight-rate mean ± sample std: ACC61.545 ±0.212, UA37.225 ±1.112,
W-F1 58.390 ±0.478. High-missing means: ACC57.501, UA31.768, W-F1 53.041.

Historical accepted three-seed reference: ACC60.693 / UA36.255 / W-F1 57.045;
new means differ by approximately +0.852 / +0.970 / +1.345pp. This is a historical
same-protocol reference, not an assertion of identical mask streams. UA does not
improve for every seed: seed68 is36.248 versus historical37.498. Retain this
negative result alongside the positive mean, and do not claim significance.

Post-completion cross-check: all24 diagnostic checkpoint/mask hashes correspond
to the training selection records; all Full ACC/UA/W-F1 values reproduce the
selected scores. Complete individual per-rate records are in RESULTS.json.
Seed66 reused; seeds67/68 from scratch. No additional HPO.

| Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|
| 66 | 61.786 | 38.435 | 58.918 |
| 67 | 61.461 | 36.991 | 58.267 |
| 68 | 61.389 | 36.248 | 57.985 |
| mean ± sample std | 61.545 ± 0.212 | 37.225 ± 1.112 | 58.390 ± 0.478 |
