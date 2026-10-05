# Gap-only Evidence Gate

INTERNAL DIAGNOSTIC ONLY

Status: deployed and QUEUED on biggpu GPU6; training has not started.
Queue process PID 70857, tmux osram_gap_only_gate_20261005.
Queued UTC 2026-10-05T03:51:29.421357+00:00; code commit 1ee8ef0.

One-stage from-scratch MOSI, 100 epochs, unchanged cfg84 causal no-JEPA Flat,
cyclic random missing 0.0–0.7, original per-rate BEST Test-oracle protocol.
Reference: osram_mosi_memory_gap_ablation_20260920/full/seed_66.
Prior four-slot Gate: osram_local_evidence_gate_20260930/results/seed_66.

Only intervention: valid Base gate fixed to 1, inactive Base/padding 0;
three active missing-modality Gap gates retain 1+0.2*tanh(shared MLP).
L2 coefficient 0.001, averaged over active-valid Gap slots only; no active Gap
means zero penalty. Gate parameter shapes and initialization unchanged.
No Memory write/read/query, Local, Flat, task head or auxiliary objective changes.
Base is not modulated and remains in the original Flat input. Joint training
can still change backbone and full-modality results.

Default flag false preserves the prior four-slot behavior and checkpoint keys.
New flag: --osram-local-evidence-gate-gap-only (requires evidence gate).

Focused CPU tests cover initialization, Base identity, masks, active-Gap
regularization normalization, no-Gap zero penalty, finite gradients and unchanged
Gap values for identical weights. No new GPU smoke or parameter sweep.
Verified on biggpu CPU with the existing s0 environment: 2 tests passed (2.99s).
Local default Python has no torch; no dependencies were installed.

GPU6 only; wait for existing core20 exclusive queue to drain. Do not interrupt it.
Remote run will save effective config, code/source hashes, reference hashes,
status, predictions and eight BEST checkpoints. No scores claimed before completion.

Remote root: /data2/yb/remote_experiments/osram_gap_only_evidence_gate_20261005.
Immutable committed snapshot: code/; live status: runs/STATUS.json; log: train.log.
Persistent runner waits for core20 DISPATCH.json to have no running/pending tasks,
then checks GPU6 UUID and at least 4000MiB free before training automatically.

Command within the remote code snapshot (existing s0 Python):

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_gap_only_evidence_gate_20261005/run.py \
--code-commit 1ee8ef0 \
--output-root /data2/yb/remote_experiments/osram_gap_only_evidence_gate_20261005/runs \
--wait-core20 /data2/yb/remote_experiments/osram_core20_20261005/runs/DISPATCH.json
```
