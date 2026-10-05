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

Status: ALL TEN COMPLETED100epochs, exit0 and outputs_verified=true. Each
retains8 BEST checkpoints,8 prediction arrays and recoverable last_training.pt.
Ordered evaluation mask hashes were verified against same-seed Flat by runner.
Dispatcher PID3935712 exited after finishing its queue; no pending work.
COMPLETION_CHECK.json archives the observed completion check; SUMMARY.json
contains all raw scores. Copying the final dispatcher JSON encountered an SSH
connection closure AFTER live completion and artifact checks succeeded; no runs
were restarted and the downloaded summary remains intact.

## Completed seed66 results

W-F1 (%). High is mean of rates.5/.6/.7; delta is percentage points vs Flat.
No seed variance/confidence intervals or significance claims from one seed.

| Category | Configuration | Mean8 | High | Mean8 delta vs Flat |
|---|---|---:|---:|---:|
|Reference|Flat|81.068|76.352|0.000|
|Reference|Old Nested64d/3layers/8nodes|80.992|76.077|-0.076|
|Ablation|Plain whole-graph GIN|80.420|75.778|-0.648|
|Ablation|No root/distance markers|79.834|75.087|-1.234|
|Ablation|No cross-role same-head edges|80.536|75.709|-0.532|
|Ablation|Last-layer summary only|80.179|75.640|-0.889|
|Sensitivity|32d|80.963|76.330|-0.106|
|Sensitivity|128d|80.198|75.519|-0.870|
|Sensitivity|1GIN layer|79.991|75.034|-1.077|
|Sensitivity|2GIN layers|80.702|75.840|-0.366|
|Sensitivity|1node per evidence|80.784|76.236|-0.285|
|Sensitivity|4nodes per evidence|80.136|75.466|-0.932|

| Rate | Flat | Old Nested | Plain | No markers | No head edges | Last layer |
|---|---:|---:|---:|---:|---:|---:|
|.0|88.205|88.078|87.398|87.073|87.733|87.153|
|.1|86.507|86.358|86.329|85.114|85.556|85.484|
|.2|83.187|83.735|82.014|81.987|82.839|81.804|
|.3|80.763|80.523|80.296|80.089|81.255|80.106|
|.4|80.827|81.012|79.987|79.147|79.781|79.967|
|.5|77.494|77.675|76.076|75.224|76.049|76.899|
|.6|75.790|75.032|75.356|75.093|75.718|75.125|
|.7|75.773|75.525|75.901|74.943|75.359|74.897|

| Rate | 32d | 128d | Depth1 | Depth2 | Groups1 | Groups4 |
|---|---:|---:|---:|---:|---:|---:|
|.0|87.914|87.278|86.962|87.616|87.667|87.451|
|.1|86.690|85.114|85.897|85.893|86.274|85.664|
|.2|83.424|82.390|82.630|82.441|83.528|82.270|
|.3|81.165|81.162|79.868|81.698|80.728|80.374|
|.4|79.519|79.088|79.472|80.444|79.365|78.931|
|.5|77.030|76.625|75.450|76.485|77.636|75.864|
|.6|76.756|75.068|74.603|75.890|75.211|74.824|
|.7|75.202|74.863|75.048|75.146|75.859|75.711|

Best NEW mean8 configuration is32d, but its mean8 is0.030points below old
Nested and0.106 below Flat. High is0.252 above old Nested and0.023 below Flat.
All four internal ablations are below old Nested on mean8 in this single seed.
This is descriptive support for retaining those components within this recipe,
not proof each is independently necessary or generally beneficial. Node counts
1/4 do not improve overall mean over8 here; no semantic head specialization
is established. No configuration exceeds Flat mean8 or high aggregate.

No automatic combined configuration search or multi-seed extension is launched.
Adaptive Test-oracle comparison remains internal, not a formal paper claim.

## Historical launch

| Host GPU | Method | PID |
|---|---|---:|
|0|nested_ab_plain_gin|3944064|
|0|nested_ab_no_markers|3969985|
|1|nested_ab_no_head_edges|4011457|
|1|nested_ab_last_layer|4085244|
|6|nested_dim32|4135716|
|6|nested_dim128|4186549|

At launch pending:depth1/depth2/groups1/groups4. LAUNCH.json is the actual dispatcher
state copied at launch, not live status; remote DISPATCH.json is authoritative.
Early plain/no-markers logs show actual optimizer epochs with jepa=0. Other
jobs are loading or training, not additional smoke tests. Do not interpret early
epoch scores as selected final results.
Dispatcher publishes DISPATCH.json plus SUMMARY.json after each admission,
including status for all ten configurations, completed per-rate W-F1,
mean8/high, parameters and peak allocated memory. Launch evidence will be
recorded only after actual process verification.
