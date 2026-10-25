# NeuMiss-only three-seed completion: verified launch

Status: **running**, not a finished performance comparison. Only NeuMiss trains;
IEMOCAP Signature is paused at the user's latest request and will not resume
automatically. PAUSE_SIGNATURE.json records preserved complete checkpoints.

- Server: `ssh biggpu` (`user22`).
- Numerical source: `e496b4842e1871b6a946dc9623ff6f2221b2828e`.
- Snapshot: `/data2/yb/remote_experiments/osram_neumiss_stage2_3seed_20261025/source_e496b48`.
- Outputs: `/data2/yb/remote_experiments/osram_neumiss_stage2_3seed_20261025/execution_v1`.
- Controller PID: `4189321`.
- tmux: `neumiss_stage2_3seed_20261025`.
- Controller log: `/data2/yb/remote_experiments/osram_neumiss_stage2_3seed_20261025/controller_v1.log`.
- Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Versions: Torch1.13.1+cu117,NumPy1.26.4,sklearn1.3.2.

## Actual finite work

Reuse completed MOSI66 fixed-rate NeuMiss eight-rate WF1 **79.61221488066607**.
New seeds67/68 each restore eight corresponding original Flat parents. Two
rate0.0 cases completed as epoch-zero exact no-ops; the remaining14 cases were
observed training at9–11/150epochs with no errors in the launch verification.
There is no seed66 retrain and no new Signature job.

Unchanged NeuMissBlock245984parameters; freeze original backbone/CHM/Flat/head.
Only current-rate train/test; no cyclic training. Original GCNet missing,
batch32,Adam WD1e-5,MSE includinggold0,LR3e-4 for100 then3e-4/10 for50 remain.
Per-rate Test-best W-F1 and same-epoch ACC/MSE remain INTERNAL TEST-ORACLE.

## Three-card bound

Only physicalGPU0/1/2 are permitted. UUIDmap is in README and live QUEUE.
Per-card cap6 with measured memory admission; all14 realtrainings run inparallel.
At verificationGPU0/1/2 used5560/5478/4340MiB respectively;GPU3 was released.
GPU4 is forbidden; unrelatedGPU5/6/7 tasks were not signalled or modified.
No method/batch or missing setting was changed to fit memory.

## Checks completed

-27 focused unit/regression tests passed locally and onremote cmerc-repro.
-All8 reusedseed66 parent/BEST/selectedprediction/mask/order/metric records
  passed independent validation before the newqueue began.
-RealNeuMiss67/miss.7 two-update GPU smoke passed; originalsavedprediction
  replay maximum absoluteerror0,initial branch equality,finite/nonzero task
  gradients,frozen-parent unchanged/no-grad,and selected checkpoint replay.
-Smoke peak371.272MiB PyTorch allocation,2.327GiB RSS,8.737s including setup,
  data/checkpoint/mask replay. These are not full training cost estimates.
-Bothnew rate0.0 summaries are complete at epoch0 with original predictions.
-All14 other workers were observed advancing epochs; no worker/controller error.

No performance conclusion is drawn before full selected-prediction validation.
Oncompletion,the controller automatically combines8reused+16new conditions into
RESULT.md/JSON/CSV andGROUPS.csv,includingallseeds/rates/highmissing means,
paired originalFlat differences andthree-seed sampleSD. Full branch/Adam/epoch/
RNG/identity last.pt and selected predictions/checkpoints stay remote. Existing
Signature and allother historical results/weights/Mask files remain preserved.
