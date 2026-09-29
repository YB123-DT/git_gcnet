# Frozen Flat history-retention diagnostic

Evaluation only; no retraining, no new module, no checkpoint re-selection.
Internal Test-oracle checkpoints are not formal paper results.

Use original cfg84 no-JEPA Flat seeds 66/67/68 and existing per-rate
checkpoints for random missing rates 0.0–0.7. Each alpha comparison within
a rate uses exactly the same checkpoint. Across rates the historical
protocol uses different selected epochs: cross-rate differences alone
cannot establish state-dependent preferences within one fixed model.

At the input of `osram.emotion_adapter`, preserve the leading Local slice
and multiply the remaining Base + masked Gap-A/T/V slices by alpha in
{0, 0.5, 1}. The Local skip is untouched. This does not rescale Memory
state, write updates, queries, keys/values, or the internal forgetting
parameter. alpha=0 is a zero-history-input intervention, not a separately
trained Local-only model. The Flat adapter biases and norm remain active.

Checks for each seed/rate:

- alpha=1 reproduces saved original W-F1 within 1e-10;
- the entire pre-intervention Flat input hash is identical for all alphas;
- evaluation mask hash and ordered labels are identical for all alphas;
- Local is unchanged; zero-valued inactive Gap slots remain zero;
- model eval mode, no_grad, all parameters have requires_grad=False.

Execution uses an isolated copy of the previously verified code on biggpu
host GPU0, UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45. Source checkpoint
root: `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full`.
Run root: `/data1/yb/remote_experiments/osram_cfg84_history_scale_20260929`.
Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`.
Dataset root: `/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.

From the isolated code root, set `CUDA_VISIBLE_DEVICES=0`,
`GCNET_DATASET_ROOT` to the above dataset root and `OMP_NUM_THREADS=2`, then
run `python -u experiments/osram_cfg84_history_scale_20260929/run.py`.
The script refuses to overwrite an existing results directory.

Successful evaluation PID: 3286205. Log: `run_verified_imports.log`.
Two earlier launches failed during imports before inference/output creation
because the new snapshot lacked config.py and legacy imported packages.
These snapshot dependencies were restored; model code was not modified.

Results include raw predictions, masks, labels, per-alpha metrics, source
hashes, configurations and a 72-row completion summary. All alpha scores
must be reported; selecting the best alpha from test labels is diagnostic
oracle analysis, not evidence of a deployable adaptive gate.
