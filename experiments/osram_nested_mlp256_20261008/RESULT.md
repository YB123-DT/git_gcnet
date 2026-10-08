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
Module parameters: old159235; wider332227 (+172992, 2.086x).
Full model expected13842020 parameters. Remote CPU checks passed: unchanged
default parameters/RNG, fixed node width/heads, zero-initialized identity,
safe inactive/padding/NaN masks, finite gradients and actual core updates.
Trainer import passed in the full sealed source archive.

Status: running. Launched2026-10-08 14:20 UTC on GPU6 with26727MiB free;
source commitc3000dd. tmux nested_mlp256_20261008, dispatcher PID2914,
training PID3345. DISPATCH.json and seed_66/train.log track live progress.
No completed performance claim yet.
