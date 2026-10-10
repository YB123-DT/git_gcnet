# Eight-rate frozen Nested correction intervention

INTERNAL DIAGNOSTIC ONLY — evaluation-only, no training.

Launched on biggpu, physical GPU7 (`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`).
PID: 3418649. Dedicated tmux socket: `gcnet_nested_deltaoff`, session: `deltaoff`.
Immutable tracked source commit: `251a545`.

Remote root: `/data1/yb/remote_experiments/osram_nested_delta_off_missing_text_20261010`.
Log: `run.log`; progress/results: `results/STATUS.json`, `results/SUMMARY.json`.
Source checkpoints: `/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66`.

```bash
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e OMP_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_nested_delta_off_missing_text_20261010/run.py \
--source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66 \
--output /data1/yb/remote_experiments/osram_nested_delta_off_missing_text_20261010/results \
--code-commit 251a545
```

All rates 0.0–0.7 use their original per-rate BEST checkpoint and original masks.
OFF retains Memory and original Local Skip; only Nested adapter-input corrections
are bypassed. Report all-test and no-Text A/V/AV groups. No-Text at rate0.0 is N/A.
Repeated samples across rates are not counted as unique independent utterances.
Status at launch: running; completion requires all eight parity/frozen-state checks.

Local correctness test: `tests/test_nested_delta_off_readout.py` — 1 passed.
