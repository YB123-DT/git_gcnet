# Launch record

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. Status: training, not completed.

- Server biggpu; host GPU6 / GPU-e4cafb17-818e-216a-b94a-7440063a9153.
- Python PID1700881; tmux socket paired_rho010_infonce_20261011, session train.
- Root: /data1/yb/remote_experiments/osram_paired_rho010_infonce_20261011.
- Immutable training implementation: 6e2a998; runner: 7fa0809.
- All four training implementation SHA256 values match completed C02 provenance.
- No resume/weight loading: one seed66 from-scratch100epoch run.
- Only configuration delta versus C02: history_contrast_weight0→.1.
- C02 source: /data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/runs/C02.
- Output: root/run; log: root/train.log. Final SUMMARY.json generated after
  checkpoint/evaluation mask and all100epoch paired training mask checks.
- Single-view inference. Eight per-rate BEST checkpoints; same Test-oracle protocol.

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_paired_rho010_infonce_20261011.run \
  --control /data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/runs/C02 \
  --output /data1/yb/remote_experiments/osram_paired_rho010_infonce_20261011/run \
  --runner-commit 7fa0809
```
