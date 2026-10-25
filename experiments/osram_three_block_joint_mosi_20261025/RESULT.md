# One-stage MOSI: execution record

Status: **formal training running**, not final results. INTERNAL TEST-ORACLE SCREEN.

Source: `96aae75cc9aecc6b9f16cc344f1feafd538740df`.
Server: `ssh biggpu`; Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Immutable source: `/data2/yb/remote_experiments/osram_three_block_joint_mosi_20261025/source_96aae75`.
Outputs: `/data2/yb/remote_experiments/osram_three_block_joint_mosi_20261025/execution_v1`.
Persistent tmux: `joint_three_mosi_20261025`; controller PID3858996.

All arms start from identical random original Flat parameters; all original
parameters train end-to-end. MOSI seed66, batch32, 150epochs, LR3e-4 first100
then3e-4/10 final50, Adam WD1e-5, original GCNet random missing and fixed accepted
test masks. No denoiser pretraining; CSDI jointly optimizes MSE + noise MSE.

| Arm | Physical GPU | Formal PID | Added parameters | Real smoke peak MiB | Selected replay max error |
|---|---:|---:|---:|---:|---:|
| Matched Flat | 0 | 3860576 | 0 | 1224.47 | 0 |
| NeuMiss | 0 | 3860577 | 245984 | 1304.46 | 0 |
| Deep Signature | 0 | 3860578 | 281856 | 1284.34 | 0 |
| CSDI joint | 1 | 3860579 | 575559 | 3304.13 | 0 |

GPU0 UUID: `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
GPU1 UUID: `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
GPU4 excluded. No other users' processes modified.

## Verified before formal training

- 28 CPU tests passed in the real remote cmerc-repro environment, including
  batch1/2/32 train/eval exact initial prediction/RNG equality, one Memory scan,
  task/denoising gradient paths, observed/padding masks, deterministic noise,
  and differentiable-checkpointed sampler equivalence.
- Four real full-batch three-update GPU smoke arms passed. Initial real batch
  predictions exactly matched their original paths; strict checkpoint loading
  replayed selected miss0.3 predictions with zero maximum absolute error.
- Finite gradients, nonzero parent/branch updates, optimizer steps and clipping
  were checked. The zero-initialized branch interior opens only after its output
  projection updates; the test does not claim nonzero upstream task gradients
  at initialization.
- Formal jobs were observed running simultaneously under the listed PIDs.

The scheduler automatically writes aggregate and per-rate RESULT.md/JSON/CSV
after all four finish150epochs, only if update counts and training/test masks
match. Until then there is no final performance claim. Remote QUEUE.json and
each train/*/SUMMARY.json are authoritative progress records.

Known limits: one seed; per-rate Test-oracle bias; CSDI adapted architecture and
its inherited short-noise schedule; fixed PCA learned from TRAIN representations
of the initial random encoder. A successful smoke is not evidence of improvement.
