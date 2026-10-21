# Original Nested GNN on current MOSEI: three-seed comparison

INTERNAL TEST-ORACLE SCREEN

This finite rerun answers the user's request to compare the **original** Nested GNN with current Flat under the accepted missing protocol. It does not restart Source Relation / 12-token experiments, perform HPO, or search new mechanisms.

## Fixed experiment

| Setting | Value |
|---|---|
| Dataset / server / environment | CMUMOSEI / `ssh biggpu` / `/data2/yb/reproduction_workspace/envs/s0/bin/python` |
| Seeds | 66, 67, 68 |
| Epochs / optimizer | 60 / Adam |
| Learning rate / weight decay | constant `1e-3` / `1e-5` |
| Batch / task | 32 conversations / original MOSI-MOSEI MSE |
| Missing | original `gcnet-random-mask-v1`, cyclic0.0–0.7, original host/guest padded-rectangle sampling |
| Main model | cfg84 no-JEPA causal OSRAM, Flat1600 and original Local Skip |
| Nested block | `nested_gnn_rooted_evidence`, dim64, depth3, hidden64, groups8, existing residual TokenAdapter |
| Excluded variants | root-aware, Local8, direct replacement, tuned dim32/depth2, Source-Role, Router, JEPA, paired views |
| Selection | per-rate TEST-best W-F1; ACC from the same prediction/mask |

Five fresh jobs are authorized: Nested66/67/68 and Flat67/68. Accepted Flat66 is reused from:

```
/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat
```

Existing Flat67/68 at LR1e-4 are **not** matched constant1e-3 controls. Historical conversation-mask-v2 values (including the earlier83.843-style score) are not current GCNet-missing baselines and must not be merged into this comparison. The worker verifies the accepted reference configuration, data, split, mask bank and original initialization rather than trusting a score alone.

Eight-rate means may use eight different selected epochs; these are not single-checkpoint deployment scores or independent-test generalization evidence. This screen does not modify the user's chosen Test-oracle protocol.

## Entry points

Run from an immutable full source snapshot containing this experiment and the existing model packages. Use **real UUIDs**, not the physical GPU numbers below:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosei_nested_gcnet_3seed_20261021.run \
  --gpus GPU-UUID-OF-HEALTHY-PHYSICAL-0,GPU-UUID-OF-HEALTHY-PHYSICAL-7 \
  --source-commit FULL_SOURCE_GIT_SHA \
  --output /data2/yb/remote_experiments/osram_mosei_nested_gcnet_3seed_20261021/execution_v1
```

Physical GPU0/7 are preferences, not assumptions of health or availability. The controller validates their current index/UUID mapping with `nvidia-smi`; physical GPU4 and UUID `GPU-46fb379f-cc90-dc82-9b5e-5d011f552264` are forbidden. Every child receives `CUDA_VISIBLE_DEVICES=<actual UUID>` and uses logical cuda:0.

The outer command should run under a project-specific persistent tmux session. Record that session, PID, full command and snapshot identity in the launch artifact. A shell returning is not proof of completed training.

### Bounded automatic queue

1. Check disk, RAM and current GPU resources.
2. Run **one** real Nested66 GPU smoke, three optimizer batches; root worker validates actual model/masks/gradients/checkpoint restoration. No redundant new Flat GPU smoke is required for unchanged original numerics.
3. Reserve Nested memory from measured peak: round up `1.5*peak+512MiB` to GiB, minimum8GiB (the first three cyclic smoke batches are not a worst-case0.7-mask guarantee); Flat reserves4GiB based on existing runs. Maintain2GiB GPU margin and at least64GiB available host RAM. Include own CPU-loading jobs in reservations before they allocate CUDA memory.
4. Launch up to four own jobs per healthy GPU, with Nested preferringGPU7 and Flat preferringGPU0. Spill only when measured admission permits. No batch or dimension reduction to fit.
5. On any worker failure, stop new admissions, retain logs, let already-running jobs finish, and mark failure. No blind retries or peer-task cancellation.
6. After completion, generate per-rate, per-seed, paired and three-seed tables. Incomplete jobs never enter final result averages.

Use `--resume` only after inspecting a genuinely stopped queue. The controller refuses a changed source/GPU/job identity and refuses recovery while a recorded worker is alive. Workers restore full `last.pt`, including optimizer/progress/RNG; loading just BEST weights is not full recovery. No queued auto-retry changes hyperparameters.

Report-only regeneration (no inference):

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python \
  -m experiments.osram_mosei_nested_gcnet_3seed_20261021.report \
  --output /data2/yb/remote_experiments/osram_mosei_nested_gcnet_3seed_20261021/execution_v1
```

## Artifacts and interpretation

Each run is isolated under `runs/CMUMOSEI/seed_<seed>/<flat|nested>/` and records effective config, source/environment/data identity, exact selected masks, matching labels/predictions, eight BEST checkpoints and full last recovery state. No protected historical checkpoint is overwritten.

Root outputs:

- `QUEUE.json`: pending/running/failed/completed state, PID, UUID, exact command, logs, memory reservations and smoke evidence.
- `REPORT.json`: completed-only metrics and reference summary hashes/provenance.
- `PER_RATE.csv`: eight rates for every completed seed/model, selected epoch and same-prediction ACC/W-F1.
- `AGGREGATE.csv`, `PAIRED_DELTA.csv`: seed-level means/high-missing means and Nested-minus-Flat differences.
- `THREE_SEED.csv`: emitted only after all three paired seeds finish; mean and **sample** standard deviation (`ddof=1`).
- `RESULT.md`: human-readable tables, state and limitations.

Weights, features and large prediction arrays stay remote and are not Git artifacts. Results must distinguish reused Flat66's older source from fresh jobs' immutable source; reviewed default-path equivalence is not a claim of byte-identical old/new source files. No positive performance conclusion follows from successful smoke or code implementation.
