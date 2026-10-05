# Old Nested: internal ablations and parameter sensitivity

INTERNAL DIAGNOSTIC ONLY. Adaptive per-rate BEST Test-oracle screening,
not validation-selected formal paper results.

## Scope and reference

Baseline commit105a838; original old Nested model commitad211c0. Existing
seed66 Flat mean8=81.068095%, high=.5/.6/.7=76.352251%; old Nested
mean8=80.992147%, high=76.077229%. Reuse both; do not retrain references.
New root-aware pooling variant is not part of this sweep.

Placement: original Flat ADAPTER INPUT ONLY, not Local Skip. Residual decoding
is zero-initialized. No change to OSRAM read/write/query, task head/loss,
random missing schedule or no-JEPA. First utterance skipped; inactive Gap
and padding safely masked. Forward512 only; zero backward half preserved.

## Locked matrix (ten fresh runs)

All variants use original mean subgraph pooling, dim64/depth3/eight nodes
per memory evidence except the SINGLE factor stated below.

| Category | Method | Change |
|---|---|---|
|Internal ablation|nested_ab_plain_gin|Whole-graph GIN instead of rooted subgraphs|
|Internal ablation|nested_ab_no_markers|Omit root/distance markers|
|Internal ablation|nested_ab_no_head_edges|Remove cross-source same-head edges|
|Internal ablation|nested_ab_last_layer|Pool last layer only instead of concatenated3layers|
|Sensitivity|nested_dim32|Node representation32|
|Sensitivity|nested_dim128|Node representation128|
|Sensitivity|nested_depth1|One GIN layer|
|Sensitivity|nested_depth2|Two GIN layers|
|Sensitivity|nested_groups1|One512-D input node per memory source|
|Sensitivity|nested_groups4|Four128-D input nodes per memory source|

OSRAM itself ALWAYS retains8heads, value64. Grouping concatenates contiguous
true-head coordinates before projection and decodes to the same512coordinates.
Grouped-node sensitivity changes token granularity AND corresponding graph
topology/parameter count; it is not a claim of isolated Memory-head causality.
Width/depth/last-layer changes alter parameter count, reported rather than hidden.
No claim of learned semantic head specialization or conflict detection.

## Protocol and artifacts

MOSI seed66,100epochs from scratch; original Adam/lr/batch32/task MSE;
cyclic missing0.0–0.7, unchanged canonical ordered evaluation masks.
Keep8 BEST checkpoints,8prediction arrays and full last_training.pt per run.
Source/data hashes and final effective config recorded. Report all failures
and negative outcomes. No automatic multi-seed expansion or combined tuning.

Server biggpu, healthy GPU0/1/6 UUID whitelist; host GPU4 forbidden. At most
two concurrent jobs per card, gradual starts with6000MiB reserved/job and
2048MiB headroom plus disk checks. Shared-GPU time is not a controlled speed benchmark.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_nested_sweep_20261005.dispatch \
  --root /data2/yb/remote_experiments/osram_nested_sweep_20261005 \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Verification:22 focused CPU tests passed on biggpu s0 in20.17s, with one
existing torch-geometric deprecation warning. Tests-first runner failure was
unsupported method; module red was missing module/registration. Default new
core exactly matches old Nested seeded parameters/RNG/output. All ten variants
have finite gradients and actual parameter updates; caller tests cover first
utterance, masks and preserved backward half. A real grouped-model task path
completed3 CPU optimizer steps with no JEPA and updated graph weights.
Separate specification and quality reviews passed. Compilation and diff checks
passed. No new dependency/GPU smoke. Existing unrelated gcnet/model.py excluded.

Measured added adapter parameters (full Flat count13,509,793):
old Nested159,235; plainGIN/no-markers158,979; no-head-edges159,235;
last-layer151,043; dim32=65,667; dim128=432,387; depth1=134,145;
depth2=146,690; groups1=158,339; groups4=158,723. Parameter count alone is
not compute cost; final metrics also retain full-model count.

Status: implementation/verification complete; launching next, no new score claimed.
Dispatcher publishes DISPATCH.json plus SUMMARY.json after each admission,
including status for all ten configurations, completed per-rate W-F1,
mean8/high, parameters and peak allocated memory. Launch evidence will be
recorded only after actual process verification.
