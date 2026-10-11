# Flat History Recency Metadata

**INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle.**

This seed-66 MOSI screen changes only the original Flat adapter input by adding
strict-past modality coverage and recency:

```text
R_t = [c_A, c_T, c_V, d_A, d_T, d_V]
```

The CHM scan, joint Block Write, Memory update, Query, Base, Gap, Local skip,
emotion normalization, task head, loss, and cyclic missing protocol are
unchanged. Training uses 100 epochs at `1e-3`, followed by a full-state resume
to epoch 150 at `1e-4`.

| Method | miss=.0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg84 Flat | 88.205 | 86.507 | 83.187 | 80.763 | 80.827 | 77.494 | 75.790 | 75.773 | 81.068 | 76.352 |
| History Recency | 86.308 | 84.462 | 82.335 | 80.478 | 79.685 | 76.028 | 74.975 | 74.941 | 79.902 | 75.315 |

History Recency changes the 8-rate mean by `-1.166` percentage points and the
high-missing mean by `-1.037` points. All selected epochs lie at or before 82,
so extending from 100 to 150 epochs did not improve any rate. This single-seed
internal screen does not support raw strict-past coverage/recency concatenation
as a useful addition to the original Flat readout.

