# Experiment manifest

## Purpose

Paired comparison of mixed-rate assignment policies for the current MOSI
Text-anchor Missing-M3 model:

1. `cyclic`: one rate per batch, cycling through eight rates;
2. `stratified`: balanced per-conversation rate assignment within each batch.

The cyclic control is inherited from:
`experiments/missing_m3_mosi_text_anchor_residual_20260904/test_oracle/`.

## Candidate provenance

- Remote run root:
  `/data2/yb/remote_experiments/missing_m3_mosi_text_anchor_residual_stratified_test_oracle_20260905`
- Remote host: `biggpu`
- Seeds: `66 67 68 69 70`
- GPUs: `2 3 5 6 7` (`GPU 4` was not used)
- Code entry point: `gcnet_missing_m3/train_gcnet.py`
- Python: `/data2/yb/reproduction_envs/s0/bin/python3.10`

## Fixed configuration

`CMUMOSI`, regression task, fold 1, 100 epochs, batch size 32, hidden size
200, `windowp=2`, `windowf=2`, `time_attention=False`, Text-anchor residual
fusion, slot representation, GraphConv second layer, no post-graph BiLSTM
ablation, latent dimension 256, four experts, top-k 2, learning rate
`1e-3`, JEPA weight `0.1`, and the official frozen feature root.

## Selection and storage

The user-requested diagnostic uses `checkpoint_selection=test-oracle` and
evaluates every epoch at all eight rates. `RESULT.md` reports the independent
best epoch for each seed/rate from `history.json`. The runner's global
mean-selected checkpoint and all eight NPZ predictions are retained for
provenance; model checkpoints are intentionally not copied into this local
results directory.

## Files

Each `results/seed_<seed>/` contains `config.json`, `history.json`,
`metrics.json`, `run.log`, and `predictions_miss_0p0.npz` through
`predictions_miss_0p7.npz`.

