# Corrected MOSEI capacity experiment: cmerc only

INTERNAL TEST-ORACLE SCREEN. This supersedes the mistakenly launched s0 batch.
The user explicitly corrected the environment. All8s0 workers and their
controller were stopped after exact PID/command/source checks, without deleting
any checkpoint, prediction, Mask, configuration or log. They are not resumed,
not relabeled as cmerc and not used as the new control scores.

Eight NEW random-initialized seed66 runs, no s0 weights loaded:

| GPU | Models |
|---:|---|
| 0 | Flat128, Flat384, Flat768, Flat256+Nested64 |
| 1 | Flat256, Flat256+Nested128, Flat512+Nested128, Flat512+Nested256 |

- `/data2/yb/conda_envs/cmerc-repro/bin/python`, Torch1.13.1+cu117,NumPy1.26.4.
- Adam, constantLR1e-4 for BOTH original and Nested parameters,WD1e-5,batch32.
-60epochs, original MSE, cyclic rates0.0-0.7, unchanged gcnet-random-mask-v1.
- Same accepted test-mask bank and UUID CUDA RNG repair as recent cmerc screen.
- Original CHM8x64, joint BlockWrite, read-before-write,write_step0.6,LocalSkip,
  output1600,task head unchanged. Nested keeps original residual/zero decoders.
- Only hidden width and existing Nested configuration change. Nested256 adds
  just a registry option for the existing dim-parametric implementation.

The worker reuses `osram_nested_cmerc_tuning_20261023.worker.main` with explicit
setup/scope callbacks. Its training/evaluation/gradient/selection/save/restore
loop and default behavior are unchanged. The custom setup uses that same
accepted cmerc loader, environment checks and RNG repair; it verifies same-width
plain/Nested original initialization and zero-decoder prediction parity.
No new global defaults, no local LR multiplier, no s0 setup, no new loss.

Reuse the immutable0867fb0 cmerc numerical snapshot and add only the Nested256
registry entry; model/sampler byte parity with the completed matched Flat is
checked. Hash callback worker/queue/report sources too. Old snapshots stay
immutable. The source/data/config/environment/RNG/Mask identities are recorded.

Compare to completed same-source paired Flat1600 (W-F1 approximately83.478),
not historical cmerc83.289 or s0 scores. Same-width256 comparisons are available
in these8arms. No cmerc plain512 exists in this budget: report its same-width
delta N/A rather than using an s0 score or silently adding a ninth run.

GPU0 UUID43d98f5a... and GPU1 UUID56b14af1...,4percard maximum. Admission
uses6GiB worker reserve (8GiB forNested256),2GiB margin,64GiB RAM/20GiB disk.
No foreign task signals, badGPU4 prohibited, no batch or precision changes.
Persistent controller fills only this finite8job queue. Full checkpoints and
exact selected predictions/Masks retained. Final report requires8newcomplete
arms plus the reused completed cmerc Flat1600; no unfinished trial ranking.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
 experiments.osram_mosei_capacity_cmerc_20261024.run \
 --source-commit RUNNER_SHA --output NEW_INDEPENDENT_OUTPUT
```

Source archive's parent contains SOURCE_MANIFEST.json. No automatic new seeds,
dataset,method or HPO expansion. Per-rate TEST-best remains internal exploratory
selection, not an independent-test or single-checkpoint generalization claim.
