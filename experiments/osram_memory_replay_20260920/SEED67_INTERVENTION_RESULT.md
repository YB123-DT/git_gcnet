# Frozen MOSI intervention: seed 67

**Internal diagnostic only; not a formal task result.**

This is a second-seed evaluation-only replay of the same frozen-write
intervention used for seed 66. It does not train, alter checkpoint weights, or
select a new epoch.

## Fixed protocol

- Dataset: CMU-MOSI, seed 67, fold 1.
- Checkpoint: `osram_mosi_memory_gap_ablation_20260920/full/seed_67/best_miss_0p7.pt`.
- Checkpoint epoch: 72; the checkpoint metadata reports the existing
  `per-rate-test-oracle` selection protocol.
- Rates: 0.1, 0.3, 0.5, 0.7.
- Modes: `reference`, `protected`, `global`.
- Device: CPU evaluation only; `evaluation_only=true`,
  `new_checkpoint_selection=false`, and `weights_unchanged=true`.
- All three modes used identical mask hashes at every rate.

`Reference` keeps the native causal OSRAM write (`eta=0.6`). `Protected`
removes the update component in historical addresses of currently missing
modalities. `Global` applies the same per-head Frobenius-norm reduction as
`Protected` without directional protection.

## Weighted-F1 and accuracy

| Rate | Reference W-F1 | Protected W-F1 | Global W-F1 | Protected − Ref. | Global − Ref. |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 85.1109 | 85.2696 | 85.1109 | +0.1587 | 0.0000 |
| 0.3 | 78.8901 | 79.0355 | 78.8901 | +0.1454 | 0.0000 |
| 0.5 | 76.6074 | 76.3289 | 76.0234 | -0.2785 | -0.5840 |
| 0.7 | 75.7389 | 75.5393 | 75.5942 | -0.1996 | -0.1447 |
| **Four-rate descriptive mean** | **79.0868** | **79.0433** | **78.9046** | **-0.0435** | **-0.1822** |

The corresponding four-rate accuracy means are 79.2302% (Reference),
79.1921% (Protected), and 79.0396% (Global).

## Interpretation

Seed 67 repeats the seed-66 direction at the aggregate level: neither frozen
intervention improves the four-rate mean. Protected write is slightly better
at 0.1/0.3 but worse at 0.5/0.7; the norm-matched Global control is also below
Reference overall and is lowest at 0.5. This does not support introducing a
dynamic gate or retraining a protection mechanism from this frozen replay.

The result is still only a two-seed diagnostic, not evidence for a final
cross-seed paper claim.

## Provenance

The copied JSON metadata and per-rate summaries are under
`mosi_seed67_intervention_allrates/`. The remote raw evaluation directory is:

`biggpu:/data2/yb/remote_experiments/osram_mosi_retention_intervention_20260920/seed67_allrates_from07/`
