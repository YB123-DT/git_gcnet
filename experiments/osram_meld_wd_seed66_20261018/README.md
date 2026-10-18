# Complete the four WD settings with seed66

Current user-selected MELD baseline (2026-10-18): LR1e-5, WD1e-5,
seeds66/67/68, ACC61.557 / UA37.728 / W-F1 58.509. See [BASELINE.md](BASELINE.md)
and [BASELINE.json](BASELINE.json) for the accepted configuration and24 protected
BEST checkpoints. Historical results remain unchanged; no new training is added.

INTERNAL TEST-ORACLE SCREEN. New budget: exactly4 from-scratch seed66 runs,
WD0/1e-5/1e-4/5e-4. All same original Flat1600 + Local Skip, Adam lr1e-5,
50epochs,batch32,constantLR,emotion-only,no added module. Missing protocol is
unchanged gcnet-random-mask-v1, per-rate testACC best with exact same-forward UA/W-F1.

The audited optimizer-screen worker and numerical b4a3138 source are reused
without editing model/trainer/sampler. Environment remains biggpu cmerc-repro,
accepted ECERC feature SHA256a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80.
The small wrapper only permits four job IDs, substitutes completed O6 seed66 as
the same-initialization/mask-stream control, and limits healthy physical GPU2
UUID GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 to4 parallel jobs. GPU4 forbidden;
existing B1/B2 onGPU0 and all unrelated jobs remain untouched.

Independent immutable source/output/queue. Save8BEST, exact best-epoch masks,
predictions and UA/ACC, plus complete last_training model/optimizer/RNG state.
No extra GPU smoke: identical model/trainer/hyperparameters already executed
for67/68; only missingseed changes. Audit actual initialization, LR/WD and all
50 train/selection mask streams against completed O6 each epoch.

After four jobs finish, reuse previous completed67/68 runs and all three WD2e-4
controls, producing five true three-seed means/sample standard deviations.
No new66 result is substituted until all50epochs and exact selection capture complete.
No overwrite of historical two-seed report or baseline checkpoints.

From the sealed snapshot, execute once through tmux:

```
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_meld_wd_seed66_20261018/run.py \
 --output /home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1 \
 --driver-commit <actual-source-commit>
```

Use explicit --resume only on this same immutable source/config/output after
checking existing queue processes. Finite queue stops after4 runs; no HPO.
Local verification: `python -m pytest tests/test_meld_wd_seed66.py
tests/test_meld_fixed_lr_wd.py tests/test_meld_optimizer_screen.py -q`.
