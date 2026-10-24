# MOSEI pure Flat256, one seed

INTERNAL TEST-ORACLE SCREEN. Exactly one additional seed66 run, no Nested,
V3, auxiliary loss, new seeds, or hyperparameter search. No MOSI job is launched.

Continue the existing MOSEI512/1024 width screen: s0 Torch2.2.2+cu121,
NumPy1.24.3, Adam LR1e-3 constant,WD1e-5,batch32,60epochs,MSE, original
gcnet-random-mask-v1 cyclic0.0-0.7 and accepted exact test masks. Only change
`osram_adapter_hidden_dim=256`; output1600,LocalSkip and CHM8x64/read-before-write/
joint BlockWrite/write_step0.6 remain unchanged. Per-rate TEST-best W-F1 uses
same-call ACC/MSE; all selected weights/predictions/masks and full last state
are preserved in an independent output directory.

Use the unchanged historical numerical snapshot
`7ec78786b12b6006bd8f6ebcfab32ee9f6026a23` from the512 grid and overlay only
this new experiment directory. The runner invokes the original validated
capacity worker, which checks data/splits/masks, untouched initialization,
one scan per batch, training input trace and counts. Expected actual parameters:
5,508,961 total,1,534,272 Flat Adapter,0 V3. These are checked at real startup.

Keep this source independent from all existing running snapshots. Physical
GPU1 is selected by its verified UUID; GPU4 is forbidden. Require at least8GiB
free VRAM,64GiB available RAM and20GiB disk before launch. No foreign process
is interrupted. No additional dedicated GPU smoke is needed for a width-only
configuration on the already validated numerical worker; the first formal
epoch supplies actual forward/backward/Adam and metric checks.

```bash
CUDA_VISIBLE_DEVICES=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
 /data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_mosei_flat256_20261024.worker --mode flat256 --phase train \
 --seed 66 --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
 --source-commit RUNNER_SHA --output NEW_INDEPENDENT_OUTPUT
```

Use tmux; `--resume` is only for full-state recovery from this same run, never
initializing from an existing BEST. After completion, RESULT.md/JSON compare
the256 arm with read-only1600/1024/512 controls. Do not mix these s0/LR1e-3
controls with later cmerc/LR1e-4 or CUDA RNG-repaired controls.
