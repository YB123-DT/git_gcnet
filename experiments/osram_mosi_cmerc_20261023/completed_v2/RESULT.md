# MOSI Flat seed66 environment control

INTERNAL TEST-ORACLE SCREEN

Same numerical source, data, config, seed, training masks and exact test masks; only environment changes.
150epochs, LR3e-4 for100 then3e-5 for50, batch32, WD1e-5, original GCNet random missing.
Fresh initialization, not continuation; no old trained weights are loaded.

| Environment | Eight-rate ACC | W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| s0 reference | 79.2683 | 79.2491 | 73.3740 | 73.4098 |
| cmerc-repro | 79.5351 | 79.4604 | 73.4756 | 73.3516 |

| Rate | s0 W-F1 | cmerc W-F1 | Delta(pp) | Selected epoch |
|---|---:|---:|---:|---:|
| 0.0 | 87.2506 | 87.6299 | +0.3793 | 40 |
| 0.1 | 83.8712 | 84.7698 | +0.8986 | 64 |
| 0.2 | 82.5951 | 82.9641 | +0.3690 | 75 |
| 0.3 | 80.7412 | 81.7781 | +1.0369 | 75 |
| 0.4 | 79.3053 | 78.4870 | -0.8183 | 45 |
| 0.5 | 75.9958 | 76.8136 | +0.8177 | 43 |
| 0.6 | 75.3279 | 73.6467 | -1.6812 | 78 |
| 0.7 | 68.9057 | 69.5945 | +0.6888 | 64 |

Eight-rate selection can combine different epochs; not single-checkpoint deployment performance.
One seed and bundled software-version change, not a causal attribution to a particular package.
No automatic extra seeds, Nested, V3 or hyperparameter search.
