# Small Flat / enlarged Nested: selected epoch audit

INTERNAL DIAGNOSTIC ONLY. MOSI seed66,100epochs, original residual connection. These combinations change both capacities relative to original Flat1600+Nested64; the enlarged-Nested subseries holds Flat256 fixed, not a full two-dimensional grid.

| Flat hidden | Nested node / MLP width | Mean8 W-F1 | Per-rate BEST epochs (.0–.7) | Median | Range |
|---:|---|---:|---|---:|---|
|256|64 / 64|80.362122|51,51,54,51,51,35,54,72|51|35–72|
|256|64 / 256|80.033360|38,51,31,57,59,61,44,57|54|31–61|
|256|64 / 512|79.883996|57,50,72,51,69,78,69,69|69|50–78|
|256|704 / 704|78.151615|76,76,45,38,62,78,78,45|69|38–78|

Sources: ../osram_small_flat_20261009/COMPLETED_RESULT.json and COMPLETED_METRICS.json in the small_flat_nested_mlp256, small_flat_nested_mlp512 and this equal_budget folder. Mean W-F1 recomputed from archived metrics agrees with reports. No inference/training or new checkpoint selection.

All selected checkpoints precede epoch79; none at100. This does not support claiming these particular runs merely ran out of training budget. No uniform monotonic shift in convergence with capacity is established. The MLP256 report's phrase “epoch75 best” refers to an earlier progress report and must not be interpreted as a selected epoch: archived selected_epoch_by_rate reaches at most61.

Separate direct/random-decoder capacity sweeps retain Flat1600 and vary Nested node/MLP widths; they are not a joint Flat-width sweep and are not mixed into this residual-series table.
