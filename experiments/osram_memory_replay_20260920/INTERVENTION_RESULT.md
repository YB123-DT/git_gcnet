# Frozen MOSI intervention: Reference / Protected / Global

**Internal diagnostic only; not a formal task result.**

This is the task-level follow-up to the retention replay. It reuses exactly one
already-trained Full MOSI checkpoint and the same deterministic masks for all
three inference-time trajectories.

## Fixed protocol

- Dataset: CMU-MOSI, seed 66, fold 1.
- Checkpoint: `osram_mosi_memory_gap_ablation_20260920/full/seed_66/best_miss_0p7.pt`.
- Checkpoint epoch: 39, selected at rate 0.7 by the existing per-rate
  Test-oracle protocol.
- Rates: 0.1, 0.3, 0.5, 0.7.
- Modes: `reference`, `protected`, `global`.
- Evaluation only: no optimizer, no checkpoint reselection, no new training.
- All mode mask hashes matched exactly; `weights_unchanged=true`.

`Reference` keeps the native causal OSRAM write. `Protected` removes the
current update component in the historical addresses of currently missing
modalities. `Global` weakens the update by the same per-head Frobenius norm as
`Protected`, but without directional protection. Thus `Global` is the
norm-matched write-weakening control.

## Weighted-F1 / accuracy

| Rate | Reference W-F1 | Protected W-F1 | Global W-F1 | Protected − Ref. | Global − Ref. |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 82.6818 | 82.6626 | 82.5162 | -0.0192 | -0.1657 |
| 0.3 | 80.4725 | 80.3269 | 80.4725 | -0.1456 | 0.0000 |
| 0.5 | 77.0410 | 76.6937 | 76.8964 | -0.3474 | -0.1447 |
| 0.7 | 75.7732 | 75.7563 | 75.3380 | -0.0169 | -0.4352 |
| **Four-rate descriptive mean** | **78.9921** | **78.8599** | **78.8058** | **-0.1323** | **-0.1864** |

Accuracy follows the same direction: four-rate mean is 79.2683% for Reference,
79.1540% for Protected and 79.0777% for Global.

## Interpretation

On this frozen checkpoint, directional protection does not improve the final
emotion metric. It is slightly below Reference at every tested rate, with the
largest drop at 0.5. The norm-matched Global control is also below Reference
on the descriptive mean and is worse at rate 0.7.

Therefore the retention damage is real, but it is not evidence that simply
removing that damage improves classification. The current write may be using
some of the same directions that are harmful to the historical probe for
useful current prediction/context. No dynamic gate or new training should be
introduced based on this frozen intervention alone.

## Provenance

The full JSON summaries and mask digests are under
`mosi_seed66_intervention_allrates/`. The remote raw evaluation directory is:

`biggpu:/data2/yb/remote_experiments/osram_mosi_retention_intervention_20260920/seed66_allrates_from07_v3/`
