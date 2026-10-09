# Small Flat plus Nested MLP512

INTERNAL DIAGNOSTIC ONLY

One additional capacity setting requested by user. Small Flat stays
4352->256->1600. Nested hidden MLPs512 (GIN64->512->64 and pool/readout192->512->64),
not512-dimensional nodes. Keep8 head nodes, node64, three layers, same rooted
topology/Tanh/residual bridges, Local Skip and task head. Memory unchanged.
No extra loss or change in original cyclic cfg84 no-JEPA training protocol.
MOSI seed66/100epochs from scratch, per-rate BEST Test-oracle with8 checkpoints
and predictions plus last_training.pt. No new seeds or additional widths.

| Nested MLP hidden | Module parameters | Whole model with small Flat |
|---|---:|---:|
|64, completed control|159235|5668196|
|256, separately running|332227|5841188|
|512, this run|562883|6071844|

Expected counts follow the unchanged architecture; the composition check
asserts the actual module count. Old small Flat+Nested mean/high W-F1:
80.362122/75.899420; small Flat79.529424/74.421931; large Flat81.068095/76.352251.
Keep the256 run intact and record all settings, not just the best one. This
is a capacity comparison, not a new method or a guarantee of improvement.
Same scalar task head1600->1; only graph internal MLP width changes against256.
Outer initialization RNG isolation preserved; graph-internal draws differ.

Reuse existing runner/dispatcher and parameterized combination check:
experiments/osram_small_flat_nested_mlp256_20261009/check.py REFERENCE_CONFIG 512.
Initial pre-change test rejected nested_mlp512 as unknown, as expected.

Server biggpu physicalGPU5 UUID GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
GPU4 forbidden. Independent sealed source/output root:
/data2/yb/remote_experiments/osram_small_flat_nested_mlp512_20261009

```text
python -m experiments.osram_nps_local_20261009.dispatch
 --method small_flat_nested_mlp512
 --root /data2/yb/remote_experiments/osram_small_flat_nested_mlp512_20261009
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
 --gpu-index 5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62
```

Existing8048MiB free-memory and26GiB free-disk admission unchanged; do not
kill unrelated processes or alter batch to fit. CPU composition check passed:
config delta, adapter1534272/module562883 counts, output1600, finite gradients,
GIN weights updated over three steps, padding output zero. Syntax/diff checks
passed. Status RUNNING, no final performance claim.
Snapshot975e484; start2026-10-09T07:34:33.694506+00:00. GPU5 admission8779MiB free.
tmux small_flat_nested_mlp512_20261009; dispatcher375273, training375362.
Live PID and effective config verified: adapter256, nested_mlp512, output1600,
seed66, epochs100. The separate256 experiment is unchanged and remains running.
