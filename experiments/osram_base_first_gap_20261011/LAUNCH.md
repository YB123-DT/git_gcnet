# Formal launch record

Status: complete; outputs verified.

- Server: biggpu (`user22`), host GPU1.
- GPU UUID: `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
- Final training PID: `3701395` (exited normally after completion).
- Code commit / immutable source: `98d6f9e`,
  `/home/qinjinghui/yb_gcnet_experiments/osram_base_first_gap_20261011/source_98d6f9e`.
- Output: `/home/qinjinghui/yb_gcnet_experiments/osram_base_first_gap_20261011/formal_98d6f9e`.
- Log: `formal_98d6f9e/train.log`; exact launch data: remote `LAUNCH.json`.
- Scope: MOSI seed66 only;100epochs at.001 then full-state50 at.0001;
  per-rate Test-oracle; no other seeds/configurations.
- Initial process check loaded all three feature streams (2199 samples each), with
  dimensions512/1024/1024. Completion requires stage150 provenance,150 history
  records, eight checkpoints/predictions, matching masks and SUMMARY.json.
- No GitHub push. Do not duplicate-launch this output; inspect recovery state first.
- Recovery note (2026-10-11 UTC): PID `3674959` was stopped after an explicit
  cancellation message, but that instruction was immediately withdrawn.  No
  other experiment was touched.  The complete `last_training.pt` at epoch 45
  was preserved, and the same run resumed with the repository recovery runner;
  epoch 46 completed under PID `3701395`.  This is a full-state continuation,
  not a fresh restart.

```bash
CUDA_VISIBLE_DEVICES=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
  /data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_base_first_gap_20261011.run \
  --root /home/qinjinghui/yb_gcnet_experiments/osram_base_first_gap_20261011/formal_98d6f9e \
  --train
```
