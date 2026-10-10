# Epoch100→150 at lower constant LR

INTERNAL DIAGNOSTIC ONLY. Both runs started; final results pending.

Implementation924d40a, pushed before deployment. Server biggpu physicalGPU7,
UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; GPU4 excluded.
Prelaunch free GPU memory19,685MiB; /data1 free94GiB. Independent copies of
checkpoint versions, no links or writes to source checkpoints.

| Model | PID | tmux | Started UTC |
|---|---:|---|---|
| Flat | 2919481 | lowlr_flat_66 | 2026-10-10T09:18:25.442079+00:00 |
| Nested | 2919558 | lowlr_nested_66 | 2026-10-10T09:18:25.443817+00:00 |

Root: `/data1/yb/remote_experiments/osram_nested_low_lr_extend150_20261010`.
Run dirs `runs/{flat,nested}_seed66`; logs `logs/{flat,nested}_seed66.log`.
Source constant100 states under
`/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs`.
Original sealed model code remains source_ad211c0, not current mutable worktree.

Both actual epoch101 optimizer groups are `[1e-4,1e-4,1e-4]`.
Restored optimizer step200 matches the original checkpoint, verified before
performing any new optimizer update. Full model/Adam moments/RNG/history/
selection/BEST references resumed; only groupLR intentionally overridden.
No cosine/warmup/architecture/loss change. Two transformation tests passed.
Configuration records and epoch101 evidence are committed alongside this file.

Command for each MODEL in flat,nested:

```sh
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 /data1/yb/remote_experiments/osram_nested_low_lr_extend150_20261010/source/run.py \
 --original /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/MODEL_seed66 \
 --historical-source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 \
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
 --output /data1/yb/remote_experiments/osram_nested_low_lr_extend150_20261010/runs/MODEL_seed66 \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --wrapper-commit 924d40a --lr .0001
```

Do not treat inherited BEST as a new improvement; report101–150-only scores and
1–150 cumulative BEST separately. Complete criteria:150history entries with
unchanged100epoch prefix,50LRtrace files, per-rate BEST/full recovery artifacts,
verified final provenance and unchanged original checkpoint hashes.
