# One-stage Evidence Gate with L1 identity regularization

User requested one-stage joint training instead of frozen two-stage Gate fitting.
Reuse cfg84 Flat, four evidence gates, gamma0.2, and lambda0.001. Change only
the previous joint-training penalty from active-valid mean `(g-1)^2` to
active-valid mean `abs(g-1)`. L2 remains the core default for old configurations.

This fixed lambda is a controlled starting point, not an optimized or proven
adequate coefficient. For a nonzero individual deviation delta the penalty
gradient magnitude ratio L1/L2 is 1/(2*abs(delta)); within the allowed0.2 range
L1 is at least2.5 times stronger at the same coefficient. At delta=0 PyTorch abs
uses subgradient0. Neither L1 nor Adam guarantees exact identity or better F1.

All original parameters train jointly, from scratch, no source checkpoint,
no freezing, no JEPA/completion/persistent mix. Original100 epochs, batch32,
Adam lr0.001/weight_decay1e-5, cyclic random-missing rates0–0.7, seeds66/67/68.
Only biggpu GPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45; max3 concurrent.
Original Flat and completed joint L2 runs are reused, not rerun. Eight best
checkpoints per seed remain **Test-oracle internal diagnostics**, not formal
validation-selected paper results.

Monitor per-evidence gate mean, saturation (<=.81 or>=1.19), mean absolute
deviation from1, and fraction within0.01 of1. Report emotion loss, raw selected
penalty and once-weighted contribution. These diagnostics are detached and do
not change the objective. L1/L2 diagnostic magnitude alone does not prove task
gradient balance or good performance; no test-label tuning is performed.

## Verification and launch plan

1. Test active-only L1 reduction/gradient, default L2 preservation and actual
   train_epoch once-weighted loss with joint original-parameter updates.
2. Run a one-epoch MOSI smoke to inspect selected penalty, magnitude and gates.
3. Snapshot code/config, launch three100-epoch tasks on biggpu GPU0, preserve
   checkpoints/provenance and automatically commit/push records.

Remote parent `/data1/yb/remote_experiments/osram_evidence_gate_l1_20260930`.
Command from its isolated code directory:

```sh
/data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_evidence_gate_l1_20260930/run.py --launch --gpus 0 --max-tasks-per-gpu 3
```
