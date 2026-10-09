# Small Flat contextual adapter: fixed three-arm comparison

INTERNAL DIAGNOSTIC ONLY

User approved adapter hidden width256, then added original Nested GNN.
Three new from-scratch MOSI seed66 runs, 100 epochs each; no sweep or automatic
multi-seed expansion. Reuse existing large-Flat references (no retraining).

| New method | Adapter | Extra mechanism |
|---|---|---|
| small_flat | 4352->256->1600 | none |
| small_flat_nps | 4352->256->1600 | original NPS, Base/Gap correction only |
| small_flat_nested | 4352->256->1600 | original residual Nested GNN |

All retain original Local Skip256->1600, final emotion_norm, task head,
Memory/query/key/value/read/write, original no-JEPA emotion-only task loss,
optimizer/lr/batch/dropout and cyclic random rates0.0-0.7. No additional view,
completion, gate or loss. NPS is NOT the Local correction or rule-width256
variant: it retains four128->96->64 rules, three steps. Nested is NOT direct,
rootaware, local8 or expanded-width; it is nested_gnn_rooted_evidence.

New switch --osram-adapter-hidden-dim (default0 means legacy output_dim).
Only two Linear shapes in emotion_adapter change; GELU, Dropout and input
LayerNorm unchanged. Last adapter Linear still zero-initialized. Construct
legacy layers first, then replace adapter under CPU fork_rng: all non-adapter
initial weights and downstream initialization RNG are preserved. Default0
retains legacy parameter keys/shapes/output. This is not checkpoint resumption.

Adapter parameters: 9535104 ->1534272 (-8000832, -83.91%).
Adapter+Skip+Norm: 9949504 ->1948672. Counts exclude the task head and any extra
mechanism; original NPS adds175168, original Nested adds159235 parameters.
Total model counts will also be checked against effective configurations.
Smaller capacity can hurt; a gain over small Flat alone is NOT a gain over
the original large Flat. No assertion that large Flat suppresses modules.

| Reused seed66 reference | Mean8 W-F1 | High W-F1 |
|---|---:|---:|
| Large Flat |81.068095|76.352251|
| Large Flat+NPS |80.981287|76.475237|
| Large Flat+Nested |80.992147|76.077229|

Output selection remains existing per-rate BEST Test-oracle. Report ACC at
the W-F1-selected checkpoint, W-F1 per rate, equal eight-rate mean and high
(.5/.6/.7) mean. Preserve eight BEST weights/predictions and last_training.pt.
Existing runner verifies evaluation-mask identity, 100 epochs, source/data
hashes and final artifacts. Not a formal paper claim.

Implementation: gcnet_missing_m3/{osram.py,model.py,train_gcnet.py}; method
mapping in experiments/osram_core20_20261005/run.py. Reuse existing persistent
NPS dispatch with healthy-GPU and disk admission, separate roots and logs.
Bounded CPU checks live in check.py; initial missing-option failure observed.

Server biggpu. GPU4 forbidden. Planned GPU5, subject to live free-memory
admission for each run; unchanged8048MiB threshold,26GiB disk requirement.
Root /data2/yb/remote_experiments/osram_small_flat_20261009.
Each method owns runs/METHOD/seed_66, DISPATCH.json and dispatcher.log.
Command from sealed source, once per method:

```text
python -m experiments.osram_nps_local_20261009.dispatch
 --method {small_flat,small_flat_nps,small_flat_nested}
 --root /data2/yb/remote_experiments/osram_small_flat_20261009/runs/METHOD
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
 --gpu-index 5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62
```

Verification passed on remote CPU: default versus pre-change OSRAM exact
state/output/RNG parity; smaller adapter keeps all non-adapter initial weights
and RNG; initial output matches legacy; three-step finite backward and actual
adapter/module updates for all three arms; padding stays zero. Trainer config
delta is only width (plus selected block), and constructed task head stays1x1600.
Measured whole-model parameters: small_flat5508961, small_flat_nps5684129,
small_flat_nested5668196. Syntax compilation and diff whitespace checks pass.
## Launch record

Status RUNNING; source snapshot a70c38d, all three on biggpu physical GPU5
(UUID above). No unrelated task stopped; admission checked separately and
starts staggered. Training process existence and RAW_CONFIG verified for all
three: hidden256/output1600, correct original block, seed66/epochs100.

| Method | Dispatcher PID | Training PID | Start UTC 2026-10-09 | Free MiB at admission |
|---|---:|---:|---|---:|
| small_flat |3943261|3943786|05:06:56|11885|
| small_flat_nps |3945844|3946154|05:07:28|10367|
| small_flat_nested |3947696|3948176|05:07:46|8645|

tmux session names: METHOD_20261009. First two confirmed epoch logs; Nested
loaded feature dimensions at initial check. Final scores and artifact checks
pending. Runtime is shared-load, not a controlled throughput benchmark.
