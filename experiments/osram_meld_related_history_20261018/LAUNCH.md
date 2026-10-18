# B1/B2 launch evidence

INTERNAL TEST-ORACLE SCREEN. Formal training is running, not completed.

- Code commit: `442596a5e247e84ada56b056eaa56c15b6c18253` (pushed).
- Server: ssh biggpu; runtime cmerc-repro (Torch1.13.1+cu117).
- Immutable snapshot:
  `/home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/source_442596a`.
- Outputs:
  `/home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/runs_v1/{B1,B2}`.
- tmux: `meld_b1b2_1018`; controller PID3141603.
- B1 PID3141613, B2 PID3141621, both concurrently on physical GPU0,
  UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; logical cuda:0.
- Full first epoch completed for both; original initialization and actual
  train/selection mask streams match O6. Eight exact BEST artifacts plus
  last_training state are present. Subsequent epochs verify masks as well.
- No unrelated task was killed, paused or moved; no GPU4 use.

Approved scope: two seed66 from-scratch50epoch runs, same O6 lr1e-5/wd2e-4
and original Flat/Local Skip, with B1/B2 sidecars only. No HPO/more seeds.
B2 train cls log includes0.05*auxiliary BCE; per-step unscaled BCE is recorded
separately in diagnostics. Evaluation CE is original task-only CE.

30 new tests and57 existing optimizer/WD checks passed (87 total). Real-batch
GPU smoke passed for both: actual backward/update, exact zero-init predictions
and RNG, original parameter initialization equality. Smoke outputs, time and
peak allocations are in SMOKE.json; these are NOT performance evidence.
The optional torch-sparse import warning exists in this fixed environment;
no environment/package was changed.

First smoke_v1 is preserved as failed: smoke-only evaluate wrapper lost its
signature, preventing capture from binding model. Regression reproduced and
fixed with functools.wraps. New smoke_v2 and formal runs use only corrected
commit442596a. No model/CHM/missing change was needed.

Launch command (cwd=immutable snapshot):

```
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_related_history_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/runs_v1 \
  --smoke-root /home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/smoke_v2
```

The finite controller persists its queue, saves per-rate ACC-best corresponding
UA/W-F1 and masks, then automatically writes RESULT.md/RESULTS.json after both
jobs finish. See LAUNCH_STATUS.json for exact commands and observed progress.
No final scores or performance improvements are claimed at launch.
