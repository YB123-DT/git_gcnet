# Small Flat plus larger Nested MLPs

INTERNAL DIAGNOSTIC ONLY

One new fixed MOSI seed66/100epoch from-scratch comparison. Keep Flat adapter
4352->256->1600. Only enlarge the Nested GIN and pooling/readout MLP hidden
widths from64 to256: GIN64->256->64; pool/readout192->256->64. Keep node width64,
8 head nodes per evidence, three GIN layers, rooted topology, Tanh, original
residual decoders, Local Skip and task head. Memory/query/write unchanged.
No loss, dropout, mask, optimizer, lr or batch changes. No extra Local split.

Reuse nested_mlp256 implementation, not a new method. Previous LARGE Flat
trial of this graph scored80.041430/75.163202, below original large Nested
80.992147/76.077229; do not hide this negative evidence. This new combination
tests the same capacity increase under small Flat, not a duplicate old run.

| Existing seed66 reference | Mean8 W-F1 | High W-F1 |
|---|---:|---:|
| Large Flat |81.068095|76.352251|
| Small Flat |79.529424|74.421931|
| Small Flat+original Nested |80.362122|75.899420|
| Large Flat+original Nested |80.992147|76.077229|
| Large Flat+wider Nested |80.041430|75.163202|

Nested parameters159235->332227 (+172992). Small Flat adapter remains1534272.
Expected whole model5841188 vs5668196 for small Flat+original Nested.
Outer OSRAM block fork_rng preserves backbone/task-head initialization. Wider
core consumes different internal draws: no claim that all tokenizer/decoder
initial values match the narrower graph. Zero-init residual semantics retained.

Protocol: unchanged original cfg84 no-JEPA, cyclic random0.0-0.7, per-rate
BEST Test-oracle (not a formal paper result). Save eight checkpoints and
predictions plus full recovery state; report ACC/W-F1, mean8 and high(.5/.6/.7).
Do not automatically launch other widths or seeds.

Server biggpu physical GPU5, UUID GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62;
GPU4 forbidden. Independent full git snapshot and output root:
/data2/yb/remote_experiments/osram_small_flat_nested_mlp256_20261009

```text
python -m experiments.osram_nps_local_20261009.dispatch
 --method small_flat_nested_mlp256
 --root /data2/yb/remote_experiments/osram_small_flat_nested_mlp256_20261009
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
 --gpu-index 5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62
```

Existing launcher keeps healthy-card/free-memory/disk checks, independent
logs and immutable-source verification. Reference scores from completed
small-Flat and nested_mlp256 reports; no reference retraining.
Remote CPU combination check passed: configuration delta limited to adapter
width and selected original graph variant; measured adapter1534272 and graph
332227 parameters; output1600, padding zero, three finite backward steps and
actual GIN weight update. Existing core semantic tests/results reused. Syntax
and diff checks pass. Initial config check rejected the unregistered method,
then passed with the new explicit mapping. No model implementation changed.
Status RUNNING. Source snapshot ee805ee; start2026-10-09T07:26:35.175721+00:00.
GPU5 had11645MiB free at admission. tmux small_flat_nested_mlp256_20261009;
dispatcher PID345788, training PID345870. Process existence and effective
RAW_CONFIG verified: adapter256, nested_mlp256, output1600, seed66, epochs100.
Independent root/seed_66/train.log and root/DISPATCH.json. No final scores yet.
