# Original no-JEPA Flat: frozen no-Local readout diagnostic

Scope: existing original cfg84 seeds66/67/68, random missing0.0–0.7,
fixed per-rate checkpoints, no retraining and no Gate/GRN enabled.
INTERNAL TEST-ORACLE EVALUATION-ONLY INTERVENTION, not a trained no-Local model.

For each checkpoint first reproduce original Flat (local_keep=1). Then:

1. Set the leading latent_dim Local slice entering emotion_adapter to zero.
2. Set the entire local_skip output to zero, including any branch bias.
3. Keep Base, masked Gap, emotion_norm and task head unchanged.

Memory Query, read/write, features and masks remain original. Current input
can still affect Memory retrieval, so this is not a history-only causal model.
The frozen task head was trained with Local present: a large drop establishes
dependence of this checkpoint, not the capacity of a retrained no-Local model.

Invariant checks: original W-F1 reproduced; pre-intervention full Flat-input
SHA256 identical on/off; ordered labels and mask hashes identical; Base/Gap
slice unchanged after intervention. All model parameters require_grad=False,
eval mode/no_grad, hooks removed after each pass. No weight edits or optimizers.

Server biggpu hostGPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
Run root /data1/yb/remote_experiments/osram_cfg84_no_local_20260929.
PID2504447; log run.log. Isolated code copied from completed history-input-gate
snapshot, with all new model switches disabled. Original checkpoints at
/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full.
Use s0 Python and GCNET_DATASET_ROOT as in the previous experiment.

Launch from isolated code root:
`CUDA_VISIBLE_DEVICES=0 GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_cfg84_no_local_20260929/run.py`.

results/ records effective source configs, hashes,48 predictions and metrics.
No existing predictions or experiment results are overwritten.
