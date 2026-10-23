# Actual real-data GPU verification

INTERNAL TEST-ORACLE SCREEN. Smoke results do not establish performance gains.

All four checks completed on `ssh biggpu`, physical GPU1,
`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`, using the unchanged s0 environment:
Torch2.2.2+cu121 / NumPy1.24.3 / sklearn1.3.2.

Immutable runtime source: `8f15f5315e1bd23fa89765d4a8344128719858e7`.

| Data/mode | Actual optimizer updates | Seconds | Peak reserved MiB | Full-state/output reload |
|---|---:|---:|---:|---|
| MOSI free | 4 | 179.772 | 1640 | exact |
| MOSI span | 4 | 180.524 | 1680 | exact |
| MOSEI free | 4 | 137.998 | 1034 | exact |
| MOSEI span | 4 | 79.982 | 1074 | exact |

Time includes setup, provenance checks, evaluation and restore, with warmed
shared caches varying by launch order. It is not an isolated forward-speed
benchmark. The measured peaks are for limited smoke batches, not a guarantee
of full-epoch peak consumption; the controller updates admission reservations
using actual formal-run peaks.

Every Query LayerNorm/Linear and output projection parameter received a finite
nonzero task gradient over these four steps. Original model parameters updated
as well. Fresh Flat predictions matched exactly across eight tested mask banks;
the original parameters/RNG and pair initialization were exactly matched.
Model/optimizer/RNG and predictions restored exactly from full saved state.

Both models contain **13,938,855** parameters, including **429,062** added
parameters and the unchanged **9,535,104**-parameter Flat Adapter.

Initial additional-module SHA256 values match within each dataset:

- MOSI free/span:
  `c5567714277de05cb6f658de33cb8633a06aeba9ee41b2391e1a5d1564491013`
- MOSEI free/span:
  `7f3a783943489718a9cb3ec7ea63c5abf6d7371deba524548621e806d00e8f6e`

Last-training-batch checks (not test-set aggregates) found the mean new query
out-of-original-space norm to be 0.978619/free versus 1.18144e-7/span on MOSI,
and 0.994226/free versus 8.69238e-8/span on MOSEI. This checks the access
constraint, not that those directions have useful sentiment information.

An additional CPU test directly supervises the added read at t=1 and verifies
finite nonzero gradients to the real t=0 Key and Value projector outputs,
without gradients to future write outputs. This passes for both modes. It is
not a performance result or a counterfactual attribution of individual writes.

Outputs are retained under:
`/data2/yb/remote_experiments/osram_conditional_reread_20261023/execution_v1/smoke/`.
No baseline or previous-run checkpoint was modified.
