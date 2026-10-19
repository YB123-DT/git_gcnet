# Single random-view F2 launch

INTERNAL DIAGNOSTIC ONLY. User requested one F2 source-decomposition model
without dual-view training, restoring random-only task loss.

- Source commit: `10699fd1274ea207b8fbb5921e50dd3d9b9412bc`.
- Branch: `feature/ecerc-features`, pushed to YB123-DT/git_gcnet via `github`.
- Remote immutable source:
  `/home/qinjinghui/remote_experiments/osram_tasr_random_only_20261019/source_10699fd`.
- Server:biggpu, original cmerc-repro runtime, no environment/data/mask changes.
- PhysicalGPU0 UUID:`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; logicalcuda:0.
- One new process, initial workerPID68231, tmux:`tasr_random_10699fd`.
- Output:
  `/home/qinjinghui/remote_experiments/osram_tasr_random_only_20261019/runs_v1/MELD_real_lr1em05`.
- Log:sibling experiment-root `train.log`.
- Original paired-view/LR experiments and other users' jobs remain running
  unchanged; no task killed, no GPU4 used, no historical artifact overwritten.

MELD seed66 only, new zero-initialized F2 Adapter, same frozen original miss=.3
Flat anchor. AdamLR1e-5/wd1e-5/batch32/30epochs. Only random-view CE with weight1.
No Persistent training view is constructed or forwarded. Persistent remains
an evaluation condition, not an extra training loss. No new seed/fold/HPO.
Original source Memory tracking and TASR architecture are unchanged.

Before launch:GPU0 free24547MiB/util0%; home50GiB free. Existing measured frozen
MELD peak below0.31GiB, with additional headroom retained. Initial observed
combined GPU0 use8908MiB/util24%, including pre-existing unrelated processes.

Validation:58 focused local TASR tests passed; one pre-existing optional
torch-sparse warning. Ten runner tests passed on the actual remote runtime.
Compileall and git diff --check passed. New tests first failed on the missing
random-only option/path, then passed; they forbid Persistent sampling and
verify one training view, unit random loss and one optimizer step.

Remote/local runner SHA256 verified equal:
`ef54668a67286c3895602c358e972d7eef8c4be0aa20ab8adbe357adc8d5d261`.

Launch is not completion. Consult SUMMARY.json and history.json for live
progress. Final evaluation automatically follows30epochs. best_adapter.pt and
last.pt preserve Adapter/optimizer/RNG/config, referencing the protected frozen
backbone. Do not call a trained output a new baseline until actually evaluated.
Do not mix this run into the original six-job LR queue or resume paired weights.
