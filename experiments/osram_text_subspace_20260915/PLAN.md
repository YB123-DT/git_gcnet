# Missing-Text predictable subspace: approved implementation plan

Goal: test a task-relevant, cross-modally predictable 32-dimensional Text target without changing causal OSRAM, MMoE, or inference.

Approved correction: `target = R(z_text_teacher.detach())`; prediction SmoothL1 updates BOTH R and Q. Never detach R's output during Stage 1. Teacher is frozen throughout. Rank 32 is not an acceptance requirement.

## Fixed design

- Stage 1: frozen supervised Teacher projectors, linear R (256→32), linear sentiment C (32→1), one shared Q MLP using fixed Audio/Visual slots and two availability bits (514→128→32). A, V, AV losses are averaged, not summed. No context inputs.
- `loss = MSE(C(R(zT)), y) + SmoothL1(Q(observed), R(zT)) + beta*(variance + covariance)`, beta=1, variance floor=1, covariance off-diagonal squared sum divided by dimension; no search. These explicit defaults are not claimed optimal.
- Full Stage-1 runner is train/validation only, selects minimum validation composite objective; default 100 epochs, existing optimizer/LR/weight decay. Smoke performs only bounded updates and cannot establish learned subspace quality.
- Stage 2: `target_space=all-modalities` preserves old behavior; `full-text` and `predictable-subspace` require original joint objective and frozen supervised Teacher. R frozen, original 256d reg/cl projected only inside loss; only missing Text contributes. Original temperature and 0.1 task weight retained.
- Stage-1 checkpoint stores selected R/C/Q weights for diagnostics, Teacher hash, selected validation epoch and config. It is not an optimizer-resume checkpoint. Stage 2 imports only R, validates Teacher hash/dimension, never imports Q/C or Student initialization.
- Old default creates no extra parameters or RNG consumption. New R initialization/loading occurs after all shared Student initialization with RNG preserved.
- No large training launches. Later Stage-2 selection remains per-seed × per-rate highest Test W-F1, internal diagnostic only.

## Execution and acceptance

- [x] Add tests first in `tests/test_text_subspace.py`: separate R sentiment/prediction gradients, Teacher stop-gradient, variance/cov statistics, Text-only masks, empty targets, both MMoE outputs differentiable through frozen R, checkpoint validation.
- [x] Run remote PyTorch tests and observe missing-feature failure before implementing `gcnet_missing_m3/text_subspace.py`.
- [x] Add model/config/CLI integration and regression tests: default exact outputs/init, frozen R, incompatible modes rejected, no auxiliary inference calls.
- [x] Add train/validation-only Stage-1 runner, explicit Stage-2 runner and checkpoint transfer. Real MOSI smoke fits a few Stage-1 updates, reports train/validation statistics, runs one Stage-2 update and verifies separate gradients and frozen hashes.
- [x] Run related OSRAM/JEPA/Teacher tests remotely, `git diff --check`, compile checks; review implementation and publish code/evidence with known gaps. Stop without five-seed training.

Remote interpreter: `/data2/yb/reproduction_envs/s0/bin/python3.10` on `biggpu`; local code synced only for changed paths. No dependencies added.
