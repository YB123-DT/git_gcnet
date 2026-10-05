# Gap-only Evidence Gate

INTERNAL DIAGNOSTIC ONLY

Status: COMPLETE; 100 epochs, canonical masks verified and eight BEST retained.
Training was on biggpu GPU6, directly concurrent with C08 as requested.
Training PID 498366, tmux osram_gap_only_gate_direct_20261005.
Started UTC 2026-10-05T03:54:51.513033+00:00; code commit 1ee8ef0.
Original queued process 70857 was terminated before training or seed output
creation; no checkpoint was removed. Only the scheduling wait was dropped.

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

GPU6 only; direct concurrent launch supersedes the original wait instruction.
Existing C08 was not interrupted. Admission free memory 18213MiB; after model
startup total GPU6 memory used 16022MiB / free 16473MiB (shared measurements).
Remote run saved effective config, code/source hashes, reference hashes,
status, predictions and eight BEST checkpoints. Local results/ contains metadata,
metrics and summary, not model weights.

## Completed seed66 results

W-F1 (%), original per-rate BEST Test-oracle; not formal validation-selected results.
The prior four-slot Gate figures are seed66, not the three-seed aggregate.

| Model | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|
| Original Flat |81.068|76.352|
| Prior Base+Gap Evidence Gate |80.357|76.292|
| Gap-only Evidence Gate |80.317|75.768|
| Gap-only minus Flat (pp) |-0.751|-0.584|
| Gap-only minus prior Gate (pp) |-0.040|-0.524|
| C08 TTT-MLP (separate core20 experiment) |80.090|75.201|

| Rate | Flat | Gap-only | Delta pp |
|---|---:|---:|---:|
|0.0|88.205|87.294|-0.911|
|0.1|86.507|86.032|-0.475|
|0.2|83.187|81.974|-1.212|
|0.3|80.763|79.933|-0.830|
|0.4|80.827|79.994|-0.833|
|0.5|77.494|77.112|-0.382|
|0.6|75.790|75.576|-0.213|
|0.7|75.773|74.616|-1.157|

Base gate is exactly 1 in all saved rate diagnostics, yet full-modality score
still declines: identity modulation does not prevent joint-training backbone
drift. This observation does not localize the cause of the performance decline.
Some rate-selected checkpoints (.3/.6/.7) have Gap gates near the .8 lower bound;
others are around .97. These are different selected checkpoints, not evidence
of a single model's rate-adaptive gate behavior. No causal mechanism claim.
No automatic multi-seed expansion, new threshold sweep or additional inference.

Remote root: /data2/yb/remote_experiments/osram_gap_only_evidence_gate_20261005.
Immutable committed snapshot: code/; live status: runs/STATUS.json;
current log: train_direct.log; original queue log: train.log.
Persistent runner checks GPU6 UUID and at least 4000MiB free before training.

Command within the remote code snapshot (existing s0 Python):

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_gap_only_evidence_gate_20261005/run.py \
--code-commit 1ee8ef0 \
--output-root /data2/yb/remote_experiments/osram_gap_only_evidence_gate_20261005/runs
```
