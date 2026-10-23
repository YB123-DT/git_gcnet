# Replay-shape numerical investigation

The first full MOSEI diagnostic (source799adcc, execution_v1) failed the
unchanged1e-6 clean-prediction check in batch4. No failed partial MOSEI score is
accepted. Old artifacts are retained; do not merge versions in final tables.

A read-only probe of the first four exact original MOSEI test batches isolated
the boundaries using ACTUAL reads sliced from the original Flat input:

| Batch (zero-based) | Full original [T,B] Flat, actual reads | Per-timestep [B] Flat, actual reads | Side replay versus actual reads |
|---|---:|---:|---:|
| 0 | 0 | 2.3842e-7 | 2.3842e-6 |
| 1 | 0 | 3.5763e-7 | 2.8610e-6 |
| 2 | 0 | 3.5763e-7 | 2.8610e-6 |
| 3 | 0 | 5.2452e-6 | 2.3842e-6 |

Both differences are float32 operation-shape effects, not different masks,
weights, Local or task definitions: full-shape Flat on original reads was
bitwise equal to the original valid predictions in all four batches.

The correction preserves production arithmetic shapes rather than relaxing
tolerance: four separate vector reads (not a four-column M@Q GEMM), followed
by one original [T,B] Flat invocation per noise draw/head. The task Jacobian
also runs on the full sequence. A zero-delta regression requires exact clean
outputs for every draw/head; synthetic tests pass. Real-data revalidation and
both full dataset reruns must use the corrected immutable source.

No CHM equation, checkpoint, noise distribution, seed, scope or Missing changes.

## Completed revalidation

Both full diagnostics completed under immutable source
`b8d4ccfef6e5824d37541fdcbf0833a0342322bf`, execution_v2.
Clean prediction replay was bitwise equal on every valid target in both rates,
including all22 MOSEI batches/rate and the previously failing batch4.
CPU and remote environment each passed all8 tests. Real GPU smoke completed
for both datasets. The old execution_v1 failure is retained, not included in
completed_v2 summaries. Neither the1e-6 tolerance nor the model was changed.
