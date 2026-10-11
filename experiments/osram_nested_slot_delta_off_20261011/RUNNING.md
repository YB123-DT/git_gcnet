# Launch record

INTERNAL DIAGNOSTIC ONLY. Started on biggpu physical GPU7, UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; process logical cuda:0.
PID1534332, tmux socket `nested_slot_delta_20261011`, session `diagnostic`.
Code commit `e31f4a2`; tracked archive extracted to isolated `source/`.
No training and no checkpoint modification. GPU4 excluded.

Remote root: `/data1/yb/remote_experiments/osram_nested_slot_delta_off_20261011`.
Log: `run.log`; progress: `results/STATUS.json`; expected final outputs:
`utterances.csv`, `per_rate.csv`, `macro.csv`, `SUMMARY.json`.

```bash
cd /data1/yb/remote_experiments/osram_nested_slot_delta_off_20261011/source
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_nested_delta_off_missing_text_20261010.run --per-slot \
--source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66 \
--output /data1/yb/remote_experiments/osram_nested_slot_delta_off_20261011/results \
--code-commit e31f4a2
```

Initial STATUS confirms running, training_runs=0. Final result is not yet claimed.
