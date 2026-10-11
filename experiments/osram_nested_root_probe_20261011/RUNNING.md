# Launch provenance

INTERNAL DIAGNOSTIC ONLY. Sealed tracked source commit b1be127. No edits to the
source directory after launch. biggpu GPU7 UUID
GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e; logical cuda:0. GPU4 forbidden.
Admission free25683MiB; existing capacity trainings untouched. PID1579314;
tmux socket nested_root_probe_20261011/session probe.
Root `/data1/yb/remote_experiments/osram_nested_root_probe_20261011`.
Log run.log; results/STATUS.json tracks rate completion. At launch rate0 train
and test extraction finished, test686 utterances and original archive parity
passed; first18 CPU probe fits started. This does not yet establish gains.

```bash
cd /data1/yb/remote_experiments/osram_nested_root_probe_20261011/source
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_nested_root_probe_20261011.run \
--source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66 \
--output /data1/yb/remote_experiments/osram_nested_root_probe_20261011/results \
--code-commit b1be127
```

Only six lightweight Probe arms train (3probe seeds x8rates=144fits), fixed100epochs.
Backbone training_runs=0. No test-label gradients, checkpoint/epoch/config selection.
Tests verified actual predecoder/root taps, first/padding/inactive root safety,
equal Probe parameter count and exact weight invariance after replacing test labels.
Expected final outputs: RESULT.md, MACRO.json, per_rate.csv, PROBE_RESULTS.json,
STATUS.json; cached features, train-only normalizers, donor indices, final Probe
weights and per-utterance predictions remain remote, not committed to Git.
