# Cyclic Signature extension: verified launch

Status: **formal training running**, not completed. No performance improvement
is claimed from launch or smoke checks.

## Immutable source and execution

- Server: `ssh biggpu` (`user22`).
- Source commit: `e0ca94c4c33de0efda2c2f4786fd289067e1ab2c`.
- Source snapshot: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
- Formal output: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/execution_v2`.
- Controller PID: `4044522`.
- Persistent tmux: `signature_cyclic_multi_v2_20261025`.
- Controller log: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/controller_v2.log`.
- Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Environment: Torch1.13.1+cu117,NumPy1.26.4,sklearn1.3.2.
- Local verification environment: `cmerc-xp-repro`; formal verification and all
  real training/smoke runs use remote `cmerc-repro`.

## Scope

MOSI66/67×8rates×150epochs; MOSEI66/67/68×8rates×60epochs; MELD66×8rates×50epochs.
Total48 formal jobs. Existing MOSI68 cyclic results are reused as prior evidence,
not rerun or overwritten. These are frozen Flat parents with only Signature
training, not48 from-scratch original Flat models. Signature LR3e-4 (MOSI drops
to3e-5 after100),batch32,Adam WD1e-5. Parent original training hyperparameters
and original GCNet missing are unchanged.

MOSI/MOSEI select current-rate TEST W-F1; MELD selects current-rate TEST ACC.
UA,ACC,W-F1 and losses are saved at the same selected epoch/mask. Internal
per-rate Test-oracle exploration only, not independent test performance.

## GPU allocation

| Physical GPU | UUID | Per-task ceiling |
| --- | --- | ---: |
| 0 | GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 |6|
| 1 | GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 |6|
| 2 | GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 |6|
| 3 | GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a |6|

UUID binding exposes only the assigned healthy card as logical cuda:0.
GPU4 is forbidden. Unrelated jobs onGPU5/6 are untouched. Concurrent ceiling24
is live VRAM/RAM-admitted and staggered; it is not a promise that24 fit under
every later resource condition. At the first verified formal snapshot there
were12 running jobs,3 perGPU,36 pending,zero failed; admission was continuing.

## Verification

40 focused/regression tests passed locally and in remote cmerc-repro. The
optional torch-sparse import warning remains; this non-graph Signature path does
not use that extension. No dependency or model setting was changed to hide it.

All three two-update real-data GPU smokes passed:

| Dataset/seed/rate | Measured allocated VRAM MiB | Peak RSS GiB | Seconds |
| --- | ---: | ---: | ---: |
| MOSI66/.7 |377.078|2.323|8.601|
| MOSEI66/.7 |285.247|3.791|27.957|
| MELD66/.7 |157.851|2.659|8.664|

These timings include setup/checkpoint/data/mask replay and are not full training
cost predictions. PyTorch allocated memory excludes CUDA context/allocator
overhead; admission also uses actual nvidia-smi free memory and a4096MiB reserve.
Smoke checks verified initial prediction equality,finite/nonzero branch
gradients,frozen parent digest/no-grad,selected branch reload equality,and
saved original prediction/mask replay. All five sentiment dataset/seed parent
config/source guards were also checked before wider admission.

Execution_v1/source_b37c873 is retained as failed preflight evidence: no formal
jobs ran, and MELD failed at data-path bootstrap. Execution_v2 fixes only import
ordering with a fresh-process regression. Old experiment results/checkpoints
were neither deleted nor overwritten.

## Monitoring and delivery

`execution_v2/QUEUE.json` records actual PIDs,commands,GPU UUIDs,logs and status.
Each formal run stores `config.json`, `history.json`, `SUMMARY.json`, selected
predictions/BEST and full `last.pt` (branch,Adam,epoch,RNG). On completion the
controller validates all48 runs and automatically writes RESULT.md/JSON/CSV.
Invalid or incomplete results are not ranked. Resume requires the same source
and explicit `--resume`; live workers are adopted rather than duplicated.
