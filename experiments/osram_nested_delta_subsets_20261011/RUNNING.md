# Launch record

INTERNAL DIAGNOSTIC ONLY. Code7218273, tracked archive in isolated remote source.
biggpu physicalGPU7 UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e; GPU4 forbidden.
Admission free26065MiB, existing two Nested capacity trainings remain untouched.
PID1569384; tmux socket nested_delta_subsets_20261011/session diagnostic.
Root `/data1/yb/remote_experiments/osram_nested_delta_subsets_20261011`.
Log run.log, progress results/STATUS.json. No training (training_runs=0).

```bash
cd /data1/yb/remote_experiments/osram_nested_delta_subsets_20261011/source
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_nested_delta_off_missing_text_20261010.run --all-subsets \
--source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66 \
--output /data1/yb/remote_experiments/osram_nested_delta_subsets_20261011/results \
--code-commit 7218273
```

All31 OFF subsets are evaluated; Full shared reference. Completion needs eight
validated rate records and all expected CSV/summary outputs. Source never modified
after launch; prior single/all-OFF diagnostic directories preserved.
