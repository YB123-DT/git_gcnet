# R12 Rank-N-Contrast three-seed confirmation

INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle BEST checkpoints.

User-authorized seeds66/67/68, MOSI, 100 epochs. Reuse completed seed66 from
`/data2/yb/remote_experiments/osram_r4_20261005/runs/R12`; do not retrain it.
New seed67/68 run from the same sealed source commit
`ff95cabe0756cf1e80af9f1552d6ecded48eddc0`, using their exact same-seed Flat
reference config. No implementation, coefficient or protocol changes.

Fixed R12: original task MSE + .1 RNC, temperature2, up to128 valid training
anchors, non-normalized Euclidean feature distance, label-distance tie handling.
Single-stage single-view cfg84 causal OSRAM + original Flat; no new inference
branch, Gate, completion or JEPA. Details in `../osram_r4_20261005/RESULT.md`.

Server biggpu; GPU6 UUID `GPU-e4cafb17-818e-216a-b94a-7440063a9153` only.
Two new independent runs may execute concurrently; no unrelated run terminated.
Expected artifacts per run: history100, metrics, eight BEST checkpoints and
predictions, last_training.pt, PROVENANCE with outputs_verified=true.

Seed66 completed: 8-rate W-F1 80.843431%, high-missing75.751701%.
Matched Flat seed66: 81.068095% / 76.352251%.
Seed67/68 both submitted on GPU6 and verified real optimizer updates:
seed67 PID3506865 (epoch4), seed68 PID3615021 (epoch1), at launch verification.
No three-seed aggregate until complete; LAUNCH.json is a historical snapshot.

Launch command (S=67 or68), from the sealed source directory:

```bash
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_core20_20261005.run --method R12 --seed S \
  --reference /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_S/config.json \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
  --output /data2/yb/remote_experiments/osram_r12_3seed_20261005/runs/seed_S \
  --gpu-uuid GPU-e4cafb17-818e-216a-b94a-7440063a9153
```

Launch-time process evidence will be recorded in LAUNCH.json; this is not a
completed-result report. All outputs remain remote, not in Git.

## Completed three-seed confirmation

All seeds completed100 epochs, PROVENANCE status=complete and
outputs_verified=true. Each seed's eight evaluation mask hashes match its
same-seed Flat reference. Seed66 was reused, not retrained.

| Seed | Flat 8-rate | R12 8-rate | Delta(pp) | Flat high | R12 high | Delta(pp) |
|---|---:|---:|---:|---:|---:|---:|
| 66 | 81.068095 | 80.843431 | -0.224664 | 76.352251 | 75.751701 | -0.600550 |
| 67 | 80.555789 | 79.711848 | -0.843941 | 75.989519 | 75.420321 | -0.569198 |
| 68 | 80.053259 | 79.559527 | -0.493732 | 74.439816 | 74.203730 | -0.236086 |
| Mean | 80.559048 | 80.038269 | -0.520779 | 75.593862 | 75.125251 | -0.468611 |

Scores are W-F1 percentages; high=.5/.6/.7. Per-rate BEST Test-oracle,
INTERNAL DIAGNOSTIC ONLY, not validation-selected formal paper results.

| Missing rate | R12 seed66 | R12 seed67 | R12 seed68 |
|---|---:|---:|---:|
| .0 | 88.140169 | 86.988104 | 87.473699 |
| .1 | 86.637300 | 85.169398 | 84.645829 |
| .2 | 83.098765 | 83.365396 | 82.177230 |
| .3 | 80.949703 | 80.299510 | 80.310250 |
| .4 | 80.666412 | 75.611412 | 79.258014 |
| .5 | 76.593235 | 75.592466 | 77.652841 |
| .6 | 75.146341 | 75.522156 | 76.321549 |
| .7 | 75.515526 | 75.146341 | 68.636801 |

The fixed transfer loses both aggregates in all three seeds. It also does not
satisfy the user's structural-Block requirement: it changes only a training
objective. Stop this route; no further RNC sweep, seeds or attempt to relabel a
loss as an architectural Block. This conclusion concerns this implemented
transfer, not a general impossibility of regression contrastive learning.
