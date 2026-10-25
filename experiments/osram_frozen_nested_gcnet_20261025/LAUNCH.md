# Verified launch receipt

- Server: `ssh biggpu`; environment `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Numerical cwd: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
- Immutable overlay: `/data2/yb/remote_experiments/osram_frozen_nested_gcnet_20261025/source_0e1bfba`.
- Source SHA: `0e1bfba6e60053cc6be88c6ed777d73812e0cff0`.
- Output: `/data2/yb/remote_experiments/osram_frozen_nested_gcnet_20261025/execution_v1`.
- Controller log: sibling `controller_v1.log`.
- tmux: `frozen_nested_gcnet_20261025`; controller PID916211.
- Started:2026-10-25T22:08:47.907408UTC.
- Eight formal jobs verified alive by exact `/proc` command/PID matching;
  progress4–19/150 at the launch receipt. No formal result is final yet.
- PhysicalGPU0/1/2, UUIDs in `LAUNCH.json`,3/3/2workers respectively;
  maximum4/card, measured capacity gating. GPU4 and other tasks untouched.

## Verification before formal admission

19 focused tests passed on the immutable remote overlay/cmerc-repro environment;
local CPU tests also passed. Compileall and diff checks passed. Spec and quality
review completed; recovery metadata/smoke recovery/cumulative resource cost were
corrected and regression-tested before snapshotting. Only new experiment files
were added; no original model or sampler was modified.

Real MOSIrate0.7 GPU smoke:2epochs,1batch each,10.700500seconds,
789.839356MiBpeak CUDA allocation,2.340260GiBpeak RSS. Added159235parameters;
full model13669028. All recorded checks passed:

- Exact initialization prediction versus accepted parent and exact saved mask.
- Original training mask/rate/update trace and exact LR expression.
- Original MSE, no auxiliary loss; original parent in eval and unchanged.
- Only Nested optimizer; connected finite gradients plus nonzero actual updates.
- One original live CHM scan per training batch.
- Strict selected block restore and all saved prediction arrays replay exactly.

Environment:Python3.10.13, Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2.
Inherited optional torch-sparse warning (`libcusparse.so.11`) remains;
the original Nested uses native tensor GIN operations, not that extension.
No dependency or environment change was made.

Smoke scores are not formal performance evidence. Completed results will be
generated automatically only after all eight150epoch runs and saved-artifact
validation. Keep this launch receipt distinct from the live remote queue state.

## Fixed comparisons and limits

Original Flat mean ACC79.535061/W-F179.460447, high W-F173.351571.
Completed NoArea frozen parallel mean W-F179.919133, serial79.847495.
New Nested uses their same own-rate parents, sampling and masks; these are
seed66 internal Test-best comparisons, not independent generalization evidence.
The mean combines eight own-rate parent/block pairs. No extra seeds or HPO.
