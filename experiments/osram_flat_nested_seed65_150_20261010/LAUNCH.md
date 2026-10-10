# Seed65 paired150 launch

INTERNAL DIAGNOSTIC ONLY. Running, results pending. No seed65 scores yet.
Code efef085, committed/pushed before deployment. Nine regression tests passed;
one pre-existing pynvml deprecation warning. Historical model source unchanged.

Server biggpu GPU7, UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e.
Root /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010.
Immutable copied runners in source/. Two independent persistent pipelines.

|Model|Stage1 PID|tmux|Effective seed|Stage1 epochs|
|---|---:|---|---:|---:|
|Flat|2999565|seed65_150_flat|65|100|
|Nested|2999643|seed65_150_nested|65|100|

Both provenance records running and PIDs alive. Stage2 runs ONLY after successful
stage1 exit. Original100 states retained under runs100; new150 under runs150.
This is constant1e-3 for100 then constant1e-4 for50, NOT constant1e-3 for150.
Stage2 restores model/optimizer/RNG/BEST state; no weight-only restart.

### flat

```sh
env CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/source/run.py --model flat --seed 65 --source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 --reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --output /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs100/flat_seed65 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --wrapper-commit efef085 --lr-schedule constant > /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/logs/flat_100.log 2>&1 && env CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/source/continue.py --original /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs100/flat_seed65 --historical-source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --output /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs150/flat_seed65 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --wrapper-commit efef085 --lr .0001 > /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/logs/flat_150.log 2>&1
```

### nested

```sh
env CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/source/run.py --model nested --seed 65 --source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 --reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --output /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs100/nested_seed65 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --wrapper-commit efef085 --lr-schedule constant > /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/logs/nested_100.log 2>&1 && env CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/source/continue.py --original /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs100/nested_seed65 --historical-source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --output /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/runs150/nested_seed65 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --wrapper-commit efef085 --lr .0001 > /data1/yb/remote_experiments/osram_flat_nested_seed65_150_20261010/logs/nested_150.log 2>&1
```

Final required checks: both complete150 with50 LR records, optimizer step200 at
epoch101, original100 prefix preserved, evaluation mask hashes equal across
models; compare cumulative BEST versus original100 and new-only101–150.
No performance claim from submission/PID alone. Failed stage1 prevents continuation.
