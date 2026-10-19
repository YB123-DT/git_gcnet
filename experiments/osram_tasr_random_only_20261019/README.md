# TASR F2 with original random-view task loss only

INTERNAL DIAGNOSTIC ONLY. One user-authorized MELD seed66 ablation,2026-10-19.

Keep the exact existing F2 source decomposition and target/source Adapter.
Remove the independent Persistent training view completely, not merely its
loss coefficient. One training forward per minibatch, one backward and one
optimizer step, with `L = CE_random` (coefficient1, not0.8).

Everything else follows the F2 LR1e-5 screen:

- Same protected MELD original Flat miss=.3 anchor and SHA256, strict loading.
- Frozen Encoder/OSRAM/Local/Flat/norm/classifier; only59,423 TASR parameters train.
- New Adapter starts from the same zero-output initialization, not a trained F2.
- Adam LR1e-5, wd1e-5,30epochs, batch32, seed66, original gradient clipping.
- Original cyclic GCNet random-mask-v1 host/guest/padding algorithm unchanged.
- Same official train/dev/test split and fixed validation/evaluation masks.
- Validation random8 ACC then CE selects one Adapter checkpoint for all tests.
- Normal random8/high and Persistent7 remain evaluation conditions only.

This is a new training protocol, not resuming or rewriting any paired-view run.
No new seed, dataset, network, loss, full-backbone training or HPO is authorized.
Main comparison: current paired F2 LR1e-5, and its identical frozen F0 anchor.
Compute is not equal: this run deliberately removes the second forward.
Historical source checkpoint is Test-oracle; this remains exploratory evidence.

New opt-in CLI `--random-only` defaultsFalse; absent it the old .8/.2 training
path is unchanged. Effective checkpoint identity records random_weight1,
persistent_weight0 and persistent_sampling disabled-random-only. Resuming with
an incompatible training identity is rejected. No second-view masks or forward
are constructed. Reports include task_forward_batches, persistent_CE=null,
zero Persistent pattern counts and null Persistent training missing fraction.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_tasr_20261018.run \
  --phase train --dataset MELD --mode real --random-only \
  --adapter-learning-rate 1e-5 \
  --output /home/qinjinghui/remote_experiments/osram_tasr_random_only_20261019/runs_v1/MELD_real_lr1em05 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

Server biggpu, cmerc-repro environment, physicalGPU0, one additional process.
Use a separate immutable Git snapshot and persistent tmux; never modify old
snapshots or protected weights. GPU4 forbidden; external tasks untouched.
Job automatically trains30epochs, saves best_adapter.pt/last.pt/history.json,
then evaluates paired F0/F2 on random8 and Persistent7, saving SUMMARY.json,
evaluation/RESULTS.json and complete predictions/masks remotely. Weights and
large predictions do not enter Git. Completion requires statuscomplete plus
all expected artifacts, not merely a running process or completed epoch.
