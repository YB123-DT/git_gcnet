# Three-seed paired confirmation: large Flat / small Flat / small Flat+Nested

INTERNAL DIAGNOSTIC ONLY

Status: four missing runs launched on biggpu, 2026-10-09 UTC; no additional architecture or seed beyond66/67/68. Results pending.

Sealed training source: commit `2d8c089`, remote `source/SNAPSHOT.json`. Persistent tmux sessions and independent dispatcher locks protect these runs from SSH disconnects and duplicate starts.

| Method | Seed | Host GPU | Training PID | tmux session |
|---|---:|---:|---:|---|
|small_flat|67|2|1260208|confirm_small_flat_67|
|small_flat|68|2|1262548|confirm_small_flat_68|
|small_flat_nested|67|3|1260440|confirm_small_flat_nested_67|
|small_flat_nested|68|3|1262555|confirm_small_flat_nested_68|

Each run root stores `DISPATCH.json` with the exact command, GPU UUID, source, PID, start time and log location. Each seed directory stores effective configuration, progress, checkpoints and predictions. Initial live-process checks passed; final scores are not yet available.

User approved the three-way confirmation, not further capacity search. Reuse completed large Flat seeds66/67/68 and small Flat / small Flat+original Nested seed66. Train only small_flat and small_flat_nested at seeds67/68,100epochs each. Flat hidden256, output1600; Nested nodes64/MLP64,3layers, original heads/topology. One-stage joint training, original MOSI regression/MSE, Adam .001/weight-decay1e-5/batch32, cyclic random missing0–.7, per-rate BEST Test-oracle to match the existing series. No frozen training, JEPA, completion, auxiliary loss or wider graph.

Reuse paths:
- Large Flat: `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_{66,67,68}`.
- Small seed66: `/data2/yb/remote_experiments/osram_small_flat_20261009/runs/{small_flat,small_flat_nested}/seed_66`.
- New roots: `/data2/yb/remote_experiments/osram_small_flat_3seed_20261009/runs/{method}_seed{seed}/seed_{seed}`.

| Model | Parameters | Seed66 mean8 / high W-F1 (%) | Seed67 | Seed68 |
|---|---:|---:|---|---|
|Large Flat|13509793|81.068095 / 76.352251|reuse complete|reuse complete|
|Flat256|5508961|79.529424 / 74.421931|new|new|
|Flat256+Nested64|5668196|80.362122 / 75.899420|new|new|

No architecture-code drift from seed66 snapshota70c38d: `git diff --name-only a70c38d HEAD -- gcnet_missing_m3 gcnet_modality_jepa` yields only the method registry, whose changes add unused variants. Current config must differ from same-seed reference only in adapter width and Nested presence. GPU2: two small_flat controls; GPU3: two small_flat_nested confirmations. Same physical serverbiggpu; GPU4 forbidden. Use isolated sealed snapshot, logs, eight checkpoints/predictions and full last_training state per run. Recheck GPU capacity at each launch; max two new jobs per card, not filling slots with unauthorized experiments.

The existing dispatcher now accepts `--seed` (default66 unchanged), uses the matching source config and output seed, and omits incomparable seed66 NPS/Nested references when running67/68.

```sh
python -m experiments.osram_nps_local_20261009.dispatch \
  --method small_flat --seed 67 \
  --root /data2/yb/remote_experiments/osram_small_flat_3seed_20261009/runs/small_flat_seed67 \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
  --gpu-index 2 --gpu-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4
```

Repeat seed68; Nested counterparts use methodsmall_flat_nested and GPU3 UUIDGPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a. No checkpoint loading: these are new from-scratch confirmations, not resumptions.
Final report must show all three seeds, per-rate ACC/W-F1, mean8/high, mean and sample SD, and paired Nested-minus-small / small-Nested-minus-large differences. Do not claim equal performance solely because a three-seed significance test is nonsignificant. No final conclusion yet.

## Progress snapshot: 2026-10-09 11:30:50 UTC

Read-only check of existing history and dispatcher records; no new training or inference launched. Values are per-rate best-so-far W-F1, not last-epoch scores. High means rates .5/.6/.7. Incomplete runs cannot establish the final three-seed comparison.

| Model | Seed | Epoch | Status | Mean8 (%) | High (%) |
|---|---:|---:|---|---:|---:|
|Flat256|67|100|complete|79.104786|74.263409|
|Flat256|68|96|running, live PID|78.855453|73.293013|
|Flat256+Nested64|67|66|running, live PID|77.880283|72.912848|
|Flat256+Nested64|68|63|running, live PID|79.091517|73.408020|

At this snapshot seed67 Nested trails the completed small Flat by1.224503pp; seed68 Nested leads the still-running small Flat by0.236064pp. These are unequal-epoch interim comparisons, not final effects. No consistent multi-seed benefit is established yet.
