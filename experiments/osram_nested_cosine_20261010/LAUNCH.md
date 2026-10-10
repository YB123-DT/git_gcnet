# Warmup/cosine paired runs

INTERNAL DIAGNOSTIC ONLY. Running, not completed at this launch record.

Implementation commit3da849d, pushed before launch. Original sealed historical
source and data hashes verified by the existing wrapper. Separate source copy:
`/data1/yb/remote_experiments/osram_nested_cosine_20261010/source`.

| Model | PID | tmux | Started UTC |
|---|---:|---|---|
| Flat | 2792290 | cosine_flat_66 | 2026-10-10T07:27:26.751153+00:00 |
| Nested | 2792380 | cosine_nested_66 | 2026-10-10T07:27:27.225006+00:00 |

Both on biggpu physicalGPU7, UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`. Checked19,685MiB free before launch;
GPU4 excluded. Independent `runs/{flat,nested}_seed66` and
`logs/{flat,nested}_seed66.log` under the experiment root.

Verified live PIDs, effective lr_schedule=cosine, warmup_ratio=.05,
learning_rate=.001; actual optimizer epoch1 rates all.0002. Both emitted their
first epoch's two training gradient records. No architecture/source/loss change.
Global constant default preserved. Four config/schedule/passive-observer tests
passed. Training results are pending; constant controls reuse completed runs.

Command for each MODEL in flat,nested:

```sh
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 /data1/yb/remote_experiments/osram_nested_cosine_20261010/source/run.py \
 --model MODEL \
 --source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 \
 --reference /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66 \
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
 --output /data1/yb/remote_experiments/osram_nested_cosine_20261010/runs/MODEL_seed66 \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --wrapper-commit 3da849d --lr-schedule cosine
```

Provenance status,100history entries,100gradient and learning-rate files and all
BEST/full-recovery artifacts jointly determine completion; PID exit alone does
not. No automatic additional seeds or alternative schedules.
