# Frozen Memory Residual Learning Experiment

INTERNAL DIAGNOSTIC ONLY

Status: completed, 168/168 planned fits. The user subsequently cancelled a proposed
seed65 Flat/Nested pair before either was launched; that pair is outside this experiment.

## Answer to the predeclared question

Under this fixed frozen-backbone / 32-hidden-unit / epoch100 correction setup,
real Memory did not yield extra task benefit. It failed to exceed Original,
Local Control, and Donor Control. Real-history scalar correction also failed
the corresponding comparison. This is negative evidence for this corrector
configuration, not proof that Memory contains no unused predictive information.

All correction groups, including Local and privileged Gold-history, have lower
eight-rate W-F1 than Original. Therefore the failure cannot establish that a
specific Memory semantic property is missing. No new mechanism or parameter
sweep was launched in response.

## Frozen source and implemented models

- Original large cfg84 no-JEPA Flat, MOSI backbone seed66. Eight source checkpoints:
  `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/best_miss_0p{0..7}.pt`.
- Exact cached outputs from the completed
  `osram_frozen_memory_audit_20261009`, whose source extraction verified original
  W-F1 and masks. Used all 686 test utterances / 656 nonneutral labels at every rate,
  and 1,284 training utterances. Train/test conversations are disjoint.
- Code snapshot: `e982f96`, sealed remote source. No original Memory, Flat, task-head,
  query, or model parameter was changed. Training only consumes frozen cached reads.
- Historical target: mean of up to three immediately preceding utterance labels,
  confined to the same conversation. Existing frozen history probes use initialization
  seed66. These probes previously used Test-oracle epoch selection.
- Each new prediction is `y0 + g(y0, Local, availability, z)`; g is
  `Linear -> GELU -> Linear`, hidden32, zero-initialized last linear weight/bias.
- A: input516, 16,577 parameters in each of Local / Donor / Real. Local has z=0;
  Memory z is the existing training-normalized, fixed-random projected 4x64 read.
- B: input262 including `has_history`, 8,449 parameters in each of the four groups.
  Donor-history and Real-history use the SAME Probe B, with unchanged current Local.
  Local-history uses the prior Probe A. Gold-history uses true past labels and is
  nondeployable, not a guaranteed numerical upper bound.
- Donors: existing fixed seed66 assignment, TRAIN only, different conversation,
  same availability and history existence. Inactive slots and first-turn history
  are zero. All groups/seeds use identical samples; no donor coverage exclusions.
- Training-only normalizers are reused. B scalars share a normalizer fitted on
  TRAIN real-history outputs, with absent history re-zeroed.

## Fixed training protocol

Adam lr0.001, weight_decay0.00001, batch128, original MSE task loss, 100 epochs;
settings inherited from the completed audit's probe runs. Three residual-head
initializations66/67/68 on ONE frozen backbone. No hyperparameter search.

The primary endpoint is the last epoch100, fixed before execution. Test labels
never enter gradients and never select a residual epoch. The inherited source
checkpoint/probe Test-oracle provenance prevents treating these scores as an
independent paper test estimate. Residual heads train on one cached mask trajectory
per rate, not newly sampled cyclic trajectories; the original backbone was trained
with cfg84 cyclic random missing. Existing history probes supply in-sample training
features; no cross-fitting was added.

## Results

W-F1 (%). Eight rates are averaged equally within each residual seed; high missing
is .5/.6/.7. Mean +/- sample SD across residual seeds, not independent backbones.
Original is one fixed reference, with no meaningful seed SD here.

| Model | Eight-rate W-F1 | High-missing W-F1 |
|---|---:|---:|
| Original Flat | 81.068 | 76.352 |
| A Local Control | 80.113 +/- 0.159 | 74.570 +/- 0.129 |
| A Donor Memory | 78.384 +/- 0.262 | 72.290 +/- 0.834 |
| A Real Memory | 77.579 +/- 0.049 | 71.464 +/- 0.509 |
| B Local-history | 80.246 +/- 0.165 | 74.933 +/- 0.158 |
| B Donor-history | 80.070 +/- 0.110 | 74.796 +/- 0.274 |
| B Real-history | 79.977 +/- 0.275 | 74.777 +/- 0.263 |
| B Gold-history, offline only | 79.973 +/- 0.219 | 75.066 +/- 0.400 |

Real Memory's eight-rate difference vs Original / Local / Donor is
-3.489 / -2.534 / -0.805 percentage points. All three residual seeds lose to
Original and Local on eight-rate W-F1. Real-history's difference vs Original /
Local-history / Donor-history is -1.091 / -0.269 / -0.093 percentage points.

For residual seed66, corrections/harms over eight rates are 267/463 for A Real
Memory and 170/244 for B Real-history. These sum repeated utterances across rates,
not unique test samples. All seeds and rates, ACC, MSE, paired differences and
correction/harm counts are in [the complete tables](summary/RESULT.md),
[per-seed CSV](summary/per_seed.csv), and [per-rate CSV](summary/per_rate.csv).

## Execution and verification

Server biggpu, CPU, two independent processes with two threads each. Started
2026-10-09 15:59:01 UTC; completed 16:04:28 UTC. Both exited0; final analysis
exited0 and verified all168 expected fits. No GPU allocation required.

```bash
cd /data2/yb/remote_experiments/osram_frozen_residual_20261009/source
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_frozen_residual_20261009.launch \
  --root /data2/yb/remote_experiments/osram_frozen_residual_20261009 \
  --audit-root /data2/yb/remote_experiments/osram_frozen_memory_audit_20261009
```

11 focused tests passed: zero-init parity, finite gradients, group capacity,
donor masking/current-Local preservation, test-label changes cannot affect fitted
weights, metric threshold/filtering, and paired summary correctness. Independently
verified72 consumed artifact hashes against the saved original audit manifests;
all unchanged. Maximum historical probe prediction replay error2.3841858e-7.
Each fit retains last.pt with optimizer/RNG state, curves and train/test predictions.
Incomplete checkpoint recovery is not automatic; the runner refuses silent overwrite.

Weights, raw caches and prediction arrays remain remotely under
`/data2/yb/remote_experiments/osram_frozen_residual_20261009/runs/`.
Small complete result tables, launch record and verification record are saved in Git.
