# Four-condition diagnostic launch

INTERNAL DIAGNOSTIC ONLY. Running, not completed at this launch record.

- Implementation commit: `14afdcb`, pushed before full launch.
- Server: biggpu; physical GPU7, `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`.
- Started UTC: 2026-10-10T06:52:10.603541+00:00.
- PID: 2756078; tmux: `history_text_context_66`.
- Root: `/data1/yb/remote_experiments/osram_history_text_context_20261010`.
- Log: `run.log`; progress/completion: `results/STATUS.json`.
- Frozen model source: `/data2/yb/remote_experiments/osram_frozen_memory_audit_20261009/source`.
  New diagnostic package is separate; original model/trainer files are not edited.
- All eight original Flat seed66 per-rate BEST checkpoints; original evaluation
  masks, strict loading, independent causal trajectories; no training/probes.
- Four local mask/contribution tests passed. Real .7 checkpoint check: original
  full-test W-F1 exactly matches stored value, four paired cases passed,
  same-current and focal-pair Local max error0, original prediction replay error
  1.7881393e-7. Frozen parameter/checkpoint hashes unchanged. This check is not a
  result about contribution reversal prevalence.

```sh
cd /data1/yb/remote_experiments/osram_history_text_context_20261010/package
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
PYTHONPATH=/data1/yb/remote_experiments/osram_history_text_context_20261010/package:/data2/yb/remote_experiments/osram_frozen_memory_audit_20261009/source \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_history_text_context_20261010.run \
  --output /data1/yb/remote_experiments/osram_history_text_context_20261010/results
```

Per-rate JSON retains all four predictions, target/focal/background identities,
mask hashes, eligibility and isolation checks. SUMMARY.json/RESULT.md are written
only after all eight rates complete. Do not relaunch into existing output paths.
