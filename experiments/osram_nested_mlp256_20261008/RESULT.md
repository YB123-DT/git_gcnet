# MOSI Nested: wider internal MLPs

INTERNAL DIAGNOSTIC ONLY

Single seed66, 100epochs. Original residual Nested, 8 memory head nodes per
evidence, node width64, depth3, rooted topology, Local Skip and zero decoders.
Only widen hidden layers inside the three GIN MLPs and pool/readout MLPs:
GIN64->64->64 becomes64->256->64; pool/readout192->64->64 becomes192->256->64.
Tanh unchanged. No attention, new loss, Memory/query/write or mask changes.
Same original cfg84 reference config and per-rate BEST protocol. Compare ACC
and W-F1, per-rate, mean8, high. Existing node-width128 and1x512 results are
different configurations; no duplicate rerun. Standard from-scratch training.

Unlike old default, widened core consumes different initialization draws;
do not claim identical shared adapter initialization. Baseline default path is
regression-checked; model's existing fork_rng isolates block initialization
from outer backbone/head RNG.

Server biggpu GPU6 only. Root:
/data2/yb/remote_experiments/osram_nested_mlp256_20261008
Command: python -m experiments.osram_nested_mlp256_20261008.dispatch --root
RUN_ROOT --data-manifest
/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
Status: pending remote checks/launch.
