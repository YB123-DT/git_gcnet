# Learning-rate and batch-size screen

**INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle checkpoint selection.**

## MOSI seed 66, 150 epochs, constant learning rate 1e-4

| Model | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat | 86.604 | 84.634 | 81.047 | 79.673 | 79.394 | 76.564 | 74.193 | 74.021 | 79.516 | 74.926 |
| Nested | 86.293 | 84.326 | 81.485 | 81.050 | 79.730 | 77.675 | 73.877 | 73.249 | 79.711 | 74.934 |
| Nested - Flat | -0.311 | -0.308 | +0.437 | +1.377 | +0.336 | +1.112 | -0.316 | -0.772 | +0.194 | +0.008 |

Selected epochs were Flat `[61, 67, 68, 103, 47, 38, 36, 86]` and Nested
`[82, 67, 55, 55, 42, 38, 61, 80]` for missing rates 0.0 through 0.7.

## MOSEI seed 66, 60 epochs

| Model/configuration | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat, batch64, LR 2e-3 | 87.477 | 86.184 | 85.814 | 84.608 | 83.359 | 82.288 | 80.836 | 79.733 | 83.787 | 80.952 |
| Flat, batch32, LR 1e-4 | 87.424 | 86.353 | 86.139 | 84.352 | 83.511 | 82.467 | 81.069 | 80.197 | 83.939 | 81.244 |
| Nested, batch32, LR 1e-4 | 87.359 | 86.370 | 86.270 | 84.178 | 83.832 | 82.662 | 81.118 | 80.376 | 84.020 | 81.385 |
| Nested - Flat at LR 1e-4 | -0.065 | +0.017 | +0.131 | -0.174 | +0.321 | +0.195 | +0.049 | +0.179 | +0.081 | +0.141 |

For context, the original seed-66 Flat run at batch32, LR 1e-3 and 100 epochs
scored 83.671 mean / 80.890 high-missing.  These comparisons confound training
duration with learning rate unless explicitly comparing the two new LR-1e-4
runs, which have matched training settings.

The raw `mask_sha256` emitted by the batch64 run differed because the hash is
sensitive to batch/chunk boundaries.  The repository's canonical ordered
sample/mask hashes match the original reference at all eight rates, so the
evaluation examples and masks themselves are unchanged.

## Related MOSEI Nested capacity screen (100 epochs, LR 1e-3)

These runs belong to `osram_mosei_nested_tuning_20261011`, not the 60-epoch
low-LR screen above.

| Configuration | Parameters | 8-rate mean | High missing |
|---|---:|---:|---:|
| Original Flat | 13,509,793 | 83.671 | 80.890 |
| Original Nested | 13,669,028 | 83.881 | 81.323 |
| Nested MLP hidden 128 | 13,726,692 | 83.985 | 81.245 |
| Nested depth 2 | 13,656,483 | 84.008 | 81.330 |
| Nested relation dim 32 + depth 2 | 13,572,259 | **84.095** | 81.279 |
| Nested dim 32 + depth 2 + MLP 64 | 13,583,651 | 83.847 | **81.422** |

The clearly parameter-increased MLP-128 model improves the overall mean by
0.104 pp over Original Nested, but reduces the high-missing mean by 0.078 pp.
The best overall configuration is smaller than Original Nested, so the screen
does not support “more parameters” as the main source of improvement.
