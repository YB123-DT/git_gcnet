# Eight-rate frozen Nested correction intervention

INTERNAL DIAGNOSTIC ONLY — evaluation-only, no training.

Launched on biggpu, physical GPU7 (`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`).
Current PID: 3423994. Dedicated tmux socket: `gcnet_nested_deltaoff`, session: `deltaoff_v2`.
Immutable tracked source commit: `1b71fca` (original snapshot plus committed run.py fix).

Remote root: `/data1/yb/remote_experiments/osram_nested_delta_off_missing_text_20261010`.
Log: `run-v2.log`; progress/results: `results-v2/STATUS.json`, `results-v2/SUMMARY.json`.
Source checkpoints: `/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66`.

```bash
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e OMP_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_nested_delta_off_missing_text_20261010/run.py \
--source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66 \
--output /data1/yb/remote_experiments/osram_nested_delta_off_missing_text_20261010/results-v2 \
--code-commit 1b71fca
```

All rates 0.0–0.7 use their original per-rate BEST checkpoint and original masks.
OFF retains Memory and original Local Skip; only Nested adapter-input corrections
are bypassed. Report all-test and no-Text A/V/AV groups. No-Text at rate0.0 is N/A.
Repeated samples across rates are not counted as unique independent utterances.
Status at launch: running; completion requires all eight parity/frozen-state checks.

Local correctness test: `tests/test_nested_delta_off_readout.py` — 1 passed.

Attempt1 failed before reporting scores: archive parity used time-major collection
instead of original conversation-major order. Its log/status are preserved under
`run.log`/`results`. Attempt2 changes only archive collection order and repeats
all parity checks. No masks or checkpoints changed. Python compilation passed.
