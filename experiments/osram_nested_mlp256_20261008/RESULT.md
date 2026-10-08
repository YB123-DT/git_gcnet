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
Completed results below supersede the launch status.

## Completed result

100/100 epochs, exit0, outputs_verified=true. All recorded output hashes
rechecked; per-rate BEST retained. No new training or inference.

|Model|Mean8 ACC|Mean8 W-F1|High ACC|High W-F1|
|---|---:|---:|---:|---:|
|flat|81.174|81.068|76.524|76.352|
|old_nested|81.059|80.992|76.169|76.077|
|mlp256|80.088|80.041|75.254|75.163|

|Rate|Original ACC|MLP256 ACC|Original W-F1|MLP256 W-F1|
|---|---:|---:|---:|---:|
|0.0|88.110|87.348|88.078|87.221|
|0.1|86.433|85.518|86.358|85.541|
|0.2|83.841|82.927|83.735|82.749|
|0.3|80.488|79.573|80.523|79.646|
|0.4|81.098|79.573|81.012|79.684|
|0.5|77.896|76.067|77.675|75.505|
|0.6|75.152|74.695|75.032|74.849|
|0.7|75.457|75.000|75.525|75.136|

MLP256 minus original residual Nested: mean8 W-F1 -0.950716pp;
high W-F1 -0.914027pp. This particular widening did not help seed66.
No automatic multi-seed expansion or further tuning. Do not infer a general
capacity/overfitting mechanism from this single run.
