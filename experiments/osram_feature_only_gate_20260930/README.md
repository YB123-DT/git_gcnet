# Feature-only gate ablation

Authorized: three seeds66/67/68,100 epochs, original cfg84 random-only no-JEPA
joint training on biggpu GPU6. Reuse original Flat and completed hierarchical
results; do not retrain references. Keep per-rate best checkpoints. All reported
comparisons remain internal Test-oracle, not formal validation-selected results.

Only intervention relative to hierarchical gate: disable Level2 computation and
set every active evidence reweight to1. Keep Level1 conditioning, initialization,
2*sigmoid feature coefficients, original Flat and Local skip unchanged. No extra
loss or frozen original backbone. Options:

```text
--osram-readout-fusion flat
--osram-hierarchical-evidence-gate
--osram-hierarchical-feature-only
```

The original 1024-dimensional readout interface consists of512 forward channels
(8x64) and512 zero backward channels. Keep its shape, do not create information in
the zero half. Thus only512 channels per active evidence have effective filtering.
Retain unused Level2 parameter tensors frozen solely to preserve exact Level1
initialization and comparable state names; they are never executed or learned.
Uniform alpha in diagnostics is bookkeeping, not a learned evidence gate;
reweight is exactly1 on active slots and0 on inactive slots.

The hypothesis is that internal feature filtering can work without cross-evidence
competition. Neither the previous negative result nor this design guarantees an
improvement. Compare all planned seeds, not the best seed.

Remote code and output under
`/data1/yb/remote_experiments/osram_feature_only_gate_20260930/{code,runs}`.
Use the existing detached GPU6 coordinator; resource checks before every child.
No changes to data, batch size, optimizer, train/test masks or checkpoint protocol.
