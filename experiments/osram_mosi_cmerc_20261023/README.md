# MOSI original Flat: cmerc-repro environment control

INTERNAL TEST-ORACLE SCREEN. One new seed66 run, not three seeds.

Only software environment changes from the accepted s0 Flat66 reference.
Use its numerical source commit `5877678ce3fac9358a94f1217e3dbac5509f6f58`,
overlaid with this independent runner. All reference numerical source hashes
are checked before training and after completion. No model files are modified.
The runner commit and numerical commit are recorded separately.

- cmerc-repro: Python3.10.13, Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2.
- Original cfg84 Flat1600 + Local Skip, no extra Block or auxiliary objective.
- Seed66,150epochs; LR3e-4 epochs1–100,3e-5 epochs101–150.
- Batch32, Adam, WD1e-5, MSE, cyclic0.0–0.7 GCNet-random-mask-v1.
- Reuse reference data/splits, initialization seed and exact evaluation banks.
- Every training epoch verifies conversation/mask trace, updates and LR against
  s0. Losses need not be numerically equal across environments.
- Record whether the initial parameter hash is exactly equal; do not pretend
  that software-version changes guarantee identical RNG initialization.
- Earliest per-rate TEST-best W-F1; ACC/MSE from the same predictions and masks.
- Full Adam/model/RNG state,8BEST,exact masks and predictions retained.
- No automatic new seeds, modules, HPO or dependency installation.

Deploy a fresh archive of the numerical commit and overlay only this directory
and its test from the runner commit. Run from that immutable composite source:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_mosi_cmerc_20261023.worker \
  --source-commit <runner-commit> --gpu-uuid <healthy-GPU-UUID> \
  --output <fresh-independent-output>
```

Run persistently in tmux. `--resume` requires the same source,environment,config
and complete `last.pt`; it does not mean starting from BEST weights.
GPU4 is prohibited. At completion RESULT.md/JSON compare the already-selected
s0 and cmerc scores without a fresh randomized evaluation.
