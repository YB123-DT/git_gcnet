# Causal OSRAM memory replay: decay versus current write

**Internal diagnostic only; not a formal task result.**

This replay uses an existing frozen **Full** MOSI checkpoint. No optimizer,
checkpoint reselection, model modification, or dynamic gate was used.

## Protocol

- Dataset: CMU-MOSI, seed 66, fold 1.
- Checkpoint: `osram_mosi_memory_gap_ablation_20260920/full/seed_66/best_miss_0p7.pt`.
- Checkpoint epoch: 39; selection protocol: `per-rate-test-oracle`, selected at
  missing rate 0.7.
- Backbone: causal OSRAM, `eta=0.6`, Flat readout, Full residual Gap,
  output 1600, 8 heads, key/value 64.
- The same frozen checkpoint was replayed at rates 0.1, 0.3, 0.5 and 0.7.
- Only cases where the current modality is missing and had a previous real
  observation are included in retention errors. `NO_HISTORY` is reported
  separately and is not assigned zero error.

For a historical real probe `(k_old, v_old)` at the current missing modality,
the diagnostic uses the exact write ordering:

\[
\bar M_t=\alpha_tM_{t-1},\qquad M_t^+=\bar M_t+\Delta M_t.
\]

`err_decay` is the relative fit error of `\bar M_t k_old` and is therefore the
write-before reference. `err_post` is the relative fit error of `M_t^+ k_old`.
The key quantity is:

\[
\text{write\_damage}=\text{err\_post}-\text{err\_decay}.
\]

Positive values mean that the current observed-modality write damaged the old
association; negative values mean that this write improved it. `decay_damage`
is reported only as a separate retention baseline.

## MOSI replay (seed 66, all heads pooled)

| Missing rate | Retention heads | Mean decay damage | Mean write damage | Write/decay | Positive write-damage |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1,656 | 0.009872 | 0.048498 | 4.91× | 75.24% |
| 0.3 | 4,264 | 0.010086 | 0.039718 | 3.94× | 73.52% |
| 0.5 | 7,128 | 0.010471 | 0.034820 | 3.33× | 73.58% |
| 0.7 | 8,616 | 0.010448 | 0.032232 | 3.08× | 74.44% |

At rate 0.7, modality-specific values are:

| Current missing modality | Heads | Mean decay damage | Mean write damage | Write/decay | Positive write-damage |
|---|---:|---:|---:|---:|---:|
| Audio | 2,896 | 0.010747 | 0.047947 | 4.46× | 75.83% |
| Text | 2,648 | 0.009906 | 0.022634 | 2.28× | 71.64% |
| Visual | 3,072 | 0.010634 | 0.025690 | 2.42× | 75.55% |

The write is not uniformly harmful: roughly one quarter of head-level writes
have negative signed damage. However, the aggregate effect is consistently
destructive on this old association probe, while the current observed write is
useful. At rate 0.7, current-write fit error falls from 0.3662 to 0.1470 for
Audio, 0.7509 to 0.3013 for Text, and 0.3940 to 0.1581 for Visual. This is a
trade-off between acquiring current evidence and retaining an unrefreshable
historical association, not a failed write solver.

At rate 0.7 the no-history fractions are 14.42% (Audio), 13.35% (Text), and
5.88% (Visual); these queries are excluded from retention-error averages.

## Existing IEMOCAP-4 five-seed replay

The already archived Full-checkpoint replay (`full5/RESULT_FULL5.md`) uses the
same `\bar M_t` reference and five seeds. Equal-modality means are:

| Missing rate | Mean decay damage | Mean write damage | Write/decay |
|---:|---:|---:|---:|
| 0.1 | 0.017147 | 0.270916 | 15.80× |
| 0.3 | 0.014214 | 0.217227 | 15.28× |
| 0.5 | 0.011769 | 0.171830 | 14.60× |
| 0.7 | 0.009830 | 0.140769 | 14.32× |

These are run-level means, not independent head-level replicates. They provide
cross-seed evidence that the separation is not specific to one MOSI seed.

## Interpretation

The memory update is **not** purely selective preservation. It successfully
fits the current observed slots, but that same block correction changes the
readout of a modality that is currently missing and cannot refresh its old
association. The average signed change is positive, while a substantial
minority of writes improve the old association. Therefore the precise claim is:

> Causal OSRAM performs a useful current-write / historical-retention trade-off;
> it can damage later-needed evidence, rather than preserving all useful
> evidence by construction.

The per-step damage is not monotonic in history distance in these records, so
we do **not** claim that every longer missing streak makes each subsequent write
worse. Any cumulative-streak claim must be based on the separate streak
analysis.

## Artifacts

- MOSI raw replay metadata and per-rate summaries:
  `mosi_seed66_allrates_from07/metadata.json` and `rate*_reference_summary.json`.
- Head-level raw records remain in the remote experiment directory and are
  mirrored as compressed JSONL under `mosi_seed66_allrates_from07/`.
- Standard coverage/distance/damage/overlap tables are under
  `mosi_seed66_allrates_from07/tables/`.
- Existing five-seed IEMOCAP evidence:
  `../osram_write_intervention_20260909/full5/RESULT_FULL5.md`.

No new dynamic gate was trained. The evidence is sufficient to justify a
direction-specific write-protection intervention as the next mechanism test,
but not to claim that such an intervention improves F1.
