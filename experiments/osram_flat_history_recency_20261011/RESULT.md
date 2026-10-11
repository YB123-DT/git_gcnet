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
to epoch 150 at `1e-4`. Results will be written after the run completes.

