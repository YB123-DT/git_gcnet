# Intermediate Flat capacity with original Nested

INTERNAL DIAGNOSTIC ONLY

Status: COMPLETED, all10 runs100/100 epochs and exit0. All200 recorded artifact hashes checked; outputs_verified for every run. Sealed code snapshot `910a3dc`; batch PID834917; tmux `medium_flat_nested_20261009`. Launch record `LAUNCH.json`; final batch/status, metrics, selected epochs, parameter counts and hashes in `COMPLETED_RESULT.json`.

## Final paired results

W-F1 (%); differences are percentage points. All runs use seed66 and per-rate Test-oracle internal selection, not formal paper evidence.

| Width | Plain mean8 | Nested mean8 | Delta | Plain high | Nested high | Delta high |
|---|---:|---:|---:|---:|---:|---:|
|384|80.137954|80.252511|+0.114557|75.107050|75.632588|+0.525538|
|512|78.991952|80.284120|+1.292168|74.197002|75.569412|+1.372410|
|768|79.846990|80.239647|+0.392657|74.875282|75.387888|+0.512605|
|1024|80.469877|79.650926|−0.818951|76.022643|75.039740|−0.982903|
|1280|79.512639|80.216063|+0.703424|74.494189|75.774818|+1.280629|

Nested improves four of five matched width controls, but enlarging Flat does not beat the existing256+Nested80.362122/75.899420 in either aggregate. Best new Nested mean8 is51280.284120 (−0.078002 vs256); best new Nested high is128075.774818 (−0.124602 vs256). These small differences are descriptive single-seed values, not significance claims.
Best overall of the ten new runs is plain102480.469877/76.022643, still below original large Flat81.068095/76.352251. The small256+Nested combination remains the lowest-parameter completed Nested combination in this comparison and is not beaten by these larger variants; this does not prove a universal optimum. No automatic new runs are launched.
Earlier very low partial results for768/1280 Nested recovered substantially: do not cite them as final failures or a diagnosed implementation bug. Full result review performed no training or new inference.

## Launch record (historical)

| Width | GPU2 plain PID | GPU3 Nested PID | Epochs at launch verification (plain/Nested) |
|---|---:|---:|---:|
|384|835943|835936|13/10|
|512|837305|837302|9/7|
|768|838678|838674|7/5|
|1024|840882|840883|2/1|
|1280|843635|843666|1/1|

At launch, verified all10 live training PIDs, effective RAW_CONFIG widths, original Nested/none presence, seed66,100epochs and unchanged regression task. Five per card, not serial completion; only initial admission was staggered.

User requests increasing Flat (not to original1600) with original Nested MLP64.
User subsequently requests GPU2/3, five simultaneous runs per GPU. Fixed ten configurations: adapter hidden384/512/768/1024/1280, each paired without/with original `nested_gnn_rooted_evidence`.
Only adapter hidden width changes: `LN(4352) -> Linear(4352,W) -> GELU -> Dropout -> Linear(W,1600)`.
Keep output1600, Local Skip, task head, causal memory/query/write and Nested graph/node64/8heads/3layers unchanged.
Nested module159235 parameters. No binary-classification switch: original MOSI regression task/MSE.
One-stage from scratch; no freezing, checkpoint initialization, extra losses, new masks or graph enlargement.
MOSI seed66,100epochs, batch32, Adam .001/weight-decay1e-5, original cyclic random missing0–.7.
For direct comparison to this existing capacity series, retain per-rate BEST Test-oracle internal screening.
No automatic further widths/seeds. Do not compare with the frozen-parent validation-selected trial.

| Adapter hidden | Adapter parameters | Whole model incl. Nested | Mean8 W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
|256, existing|1534272|5668196|80.362122|75.899420|
|384, new|2296256|6430180|80.252511|75.632588|
|512, new|3058240|7192164|80.284120|75.569412|
|768, new|4582208|8716132|80.239647|75.387888|
|1024, new|6106176|10240100|79.650926|75.039740|
|1280, new|7630144|11764068|80.216063|75.774818|
|1600, existing original|9535104|13669028|80.992147|76.077229|

Parameter formulas: adapter5953W+10304; other parameters incl. original Nested4133924.
No-Nested controls have159235 fewer parameters at each width. Existing256/1600 results reused, not retrained.
This measures both the performance/parameter tradeoff and incremental Nested effect against each same-width control. Single-seed results cannot establish robustness. No new losses or modifications besides adapter width/Nested presence.

Reuse `osram_core20_20261005.run` plus `osram_nps_local_20261009.dispatch`; no model implementation changes.
Configuration tests first reject unknown experiment names; after registration check only two config deltas, counts, unchanged original Nested/outer RNG, padding and three finite update steps.
Keep existing small_flat* names mapped to256 unchanged.
Measured CPU checks for all five widths passed: exact adapter/module parameter counts, original Nested and non-adapter initialization unchanged, outer RNG unchanged, padding zero, finite gradients and actual adapter/Nested updates over three steps. Plain-control configuration deltas contain only adapter width; old small_flat names still map to256.

Server biggpu, physicalGPU2 UUID GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 and GPU3 UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a; GPU4 forbidden. GPU2 five plain controls; GPU3 five Nested runs. Same device model, no speed-comparison claims under concurrency.
One sealed source, ten isolated output roots. Staggered admission to observe memory before each additional pair; max5 per GPU, no batch changes.
Root for each method: `/data2/yb/remote_experiments/osram_medium_flat_nested_20261009/runs/METHOD`.

```sh
python -m experiments.osram_nps_local_20261009.dispatch \
  --method flat512_nested \
  --root /data2/yb/remote_experiments/osram_medium_flat_nested_20261009/runs/flat512_nested \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
  --gpu-index 3 --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

Run each of the five widths; plain controls use `flatW`, GPU2/its UUID. Save eight BEST checkpoints/predictions and complete last_training state. Report all ten completed/failed configurations, per-rate ACC/W-F1, mean8/high and parameter counts; no performance claim from initial checks.

Batch entry: `python -m experiments.osram_medium_flat_nested_20261009.launch --root /data2/yb/remote_experiments/osram_medium_flat_nested_20261009`. Waits for first train/eval epoch of each pair before submitting the next; individual launchers enforce available GPU memory/disk and UUID checks. Tracks `BATCH.json` and each method's `DISPATCH.json`. Does not wait for completion of a run before starting the next pair.
