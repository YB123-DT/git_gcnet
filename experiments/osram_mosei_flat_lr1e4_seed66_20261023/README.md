# Complete MOSEI original Flat constant1e-4 seeds with seed66

INTERNAL TEST-ORACLE SCREEN. Exactly ONE new seed66 training run; reuse completed
67/68 read-only. No Nested/V3/Router/new module or auxiliary loss. This is not
authorization to run the separately discussed Flat/Nested capacity experiment.

Original cfg84 no-JEPA Flat1600/Local Skip/8heads×64, joint Block Write0.6;
new gcnet-random-mask-v1/cyclic0–.7, official MOSEI split/features, original s0
Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2. Adam/batch32/WD1e-5, constantLR1e-4,
60epochs, original MSE/dropout and per-rate earliest TEST-best W-F1. Preserve
eightBEST, exact same-call predictions/masks and full model/Adam/RNG last.

Reuse the existing low-LR worker. The only extension is accepting the newly
authorized seed66; its old67/68 controller remains unchanged. Newly introduced
DISABLED/default config fields may be reconciled with the old reference schema;
non-default changes must fail. No CHM/model/sampler computation is modified.

Existing same-seed complete source-role training supplies only initialization,
mask/order/update and split checks; its trained weights are NEVER loaded.
Each formal epoch checks its original mask trace/update count; LR stays1e-4.
Use existing original-Flat GPU validation rather than a redundant new training
smoke. Verify the real formal first epoch's finite gradients/LR/update record.

biggpu physicalGPU7, UUID`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e` initially
healthy/free. GPU4 forbidden. Require64GiB host headroom,12GiB disk,6GiB GPU;
do not stop others or change batch. Independent immutable snapshot/output,
tmux persistence, explicit same-source resume only. No extra seeds/configs.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_mosei_flat_lr1e4_seed66_20261023.run \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --source-commit IMMUTABLE_SHA --output /fresh/output
```

After the new60-epoch worker completes, the controller verifies all three
seeds' selected artifact hashes and generates mean/high/per-rate ACC/W-F1/MSE
and sample SD automatically. It does not label unfinished runs completed.
Full weights/data remain on biggpu; only code/config/small records go to Git.

## Nested capacity lookup

No completed MOSEI new-Missing smallFlat+largeNested run was found. Current
original Nested keeps Flat1600 fixed (three-seed W-F1 Flat83.241/Nested83.198).
Old MOSEI tuning changed only Nested dim/depth/MLP/groups, not Flat width.
Actual older smallFlat+largeNested experiments were MOSI under old Missing.
MOSEI new-Missing smallFlat+largeV3 is V3, not Nested. Evidence will be preserved
separately; no old-mask score is mixed into current comparisons.
