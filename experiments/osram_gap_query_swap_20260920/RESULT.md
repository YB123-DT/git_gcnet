# Frozen Gap-query identity swap on MOSI

**Internal diagnostic only; not a formal task result.**

This evaluation reuses the seed-67 Full causal OSRAM checkpoint and the
existing deterministic test masks. It changes only the read-side identity of
the two simultaneously active Gap queries. Keys, values, availability masks,
Base query, Local path, readout parameters, and every persistent memory write
remain unchanged.

## Protocol

- Dataset: CMU-MOSI, seed 67, fold 1.
- Checkpoint: `osram_mosi_memory_gap_ablation_20260920/full/seed_67/best_miss_0p7.pt`.
- Checkpoint epoch: 72; existing selection protocol: `per-rate-test-oracle`.
- Rates: 0.1, 0.3, 0.5, 0.7.
- Reference: native causal OSRAM (`eta=0.6`).
- Swapped: when exactly one modality is observed, exchange the two missing
  modality query vectors (A/T/V slots); all other rows are unchanged.
- Evaluation only: no optimizer, no checkpoint reselection, no new training.

The two runs have identical post-write memory tensors at every time step for
every rate (`memory_trajectory_equal=true`). Thus the comparison isolates
read-side query identity rather than a changed memory trajectory.

## Overall and affected-utterance W-F1

`two-missing` means the current utterance has exactly one observed modality,
so the two missing-modality Gap queries are simultaneously active and swapped.

| Rate | Affected rows | Reference overall | Swapped overall | Δ overall | Reference two-missing | Swapped two-missing | Δ affected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 18 / 686 | 85.1109 | 85.1109 | 0.0000 | 77.7778 | 77.7778 | 0.0000 |
| 0.3 | 160 / 686 | 78.8901 | 78.8901 | 0.0000 | 68.3022 | 68.3022 | 0.0000 |
| 0.5 | 355 / 686 | 76.6074 | 76.3155 | -0.2919 | 69.6028 | 69.0465 | -0.5562 |
| 0.7 | 526 / 686 | 75.7389 | 75.9012 | +0.1623 | 75.4524 | 75.6642 | +0.2117 |

The overall changes are small and not monotonic across rates. The reference
and swapped predictions differ numerically on the affected rows, but most
changes do not cross the sentiment sign decision boundary.

## Which missing pair is affected?

Availability is shown as A/T/V. The rows below report W-F1 on nonzero-label
examples within each pair:

| Observed modality | Missing query pair | 0.1 Δ | 0.3 Δ | 0.5 Δ | 0.7 Δ |
|---|---|---:|---:|---:|---:|
| V (`001`) | A ↔ T | 0.0000 | 0.0000 | 0.0000 | +0.6270 |
| T (`010`) | A ↔ V | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A (`100`) | T ↔ V | 0.0000 | 0.0000 | -1.6971 | 0.0000 |

This first seed therefore gives no evidence that query identity is a uniform
source of the Gap gain. It is most consistent with a weak, rate- and
pattern-dependent role: the logits move, but the classification decisions are
usually unchanged.

## Artifacts

- Per-rate summaries and affected-utterance JSONL:
  `mosi_seed67_allrates/`.
- Remote raw evaluation directory:
  `biggpu:/data2/yb/remote_experiments/osram_mosi_gap_query_swap_20260920/seed67_allrates_v2/`.
