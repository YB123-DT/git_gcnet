# Formal launch record

Status: running; not a final result.

- Server: biggpu (`user22`).
- Host GPU:0, healthy; host GPU4 explicitly excluded.
- GPU UUID: `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
- Code commit / immutable source: `9fc6feb`,
  `/home/qinjinghui/yb_gcnet_experiments/osram_counterfactual_readouts_20261011/source_9fc6feb`.
- Output root:
  `/home/qinjinghui/yb_gcnet_experiments/osram_counterfactual_readouts_20261011/formal_9fc6feb`.
- Marginal PID/log: `3904674`, `formal_9fc6feb/marginal.log`.
- Coalition PID/log: `3905138`, `formal_9fc6feb/coalition.log`.
- Both independent processes are concurrently bound to the same host GPU0 UUID.
- Observed live allocation after both entered training: marginal2,874MiB,
  coalition4,478MiB; both completed epoch1 without non-finite failure.
- Scope: MOSI seed66 only;100epochs at.001 then full-state50 at.0001;
  cyclic random missing; per-rate Test-oracle; no other seeds/configurations.
- Each run has an isolated directory, lock, provenance, log, checkpoints,
  predictions and full `last_training.pt` recovery state.
- Completion requires each provenance to report `status=complete` and
  `outputs_verified=true`,150 history records, eight best checkpoints and
  predictions, matching mask hashes, then a combined `SUMMARY.json`.
- No GitHub push. Do not duplicate-launch either task; inspect its recovery
  state first if a PID disappears.

Launch commands used the same pattern, with `--train 0` for marginal and
`--train 1` for coalition:

```bash
CUDA_VISIBLE_DEVICES=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  /data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_counterfactual_readouts_20261011.run \
  --root /home/qinjinghui/yb_gcnet_experiments/osram_counterfactual_readouts_20261011/formal_9fc6feb \
  --train INDEX \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```
