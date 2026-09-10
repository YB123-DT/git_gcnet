# Move / Swap controls: role exchange is tolerated, single-donor behavior is asymmetric

MOSI, seeds 66–70, frozen causal Flat eta=.6, exactly-one-missing AT/AV/TV
utterances. Local is unchanged in every intervention. No training, model edits,
normalization, epoch reselection or representation-geometry analysis.

The entries below average eligible-rate W-F1 (.1–.7) within each seed, then seeds.
Rate 0 has no eligible samples. These are **subset descriptive means**, not full-test
eight-rate benchmark scores. W-F1 excludes zero labels, following the original metric.

| Mode | Base slot, unique active Gap slot | W-F1 (%) |
|---|---|---:|
| Normal | B, G | 79.182 |
| Base-only | B, 0 | 78.352 |
| Gap-only | 0, G | 78.119 |
| Move B→Gap | 0, B | 78.969 |
| Move G→Base | G, 0 | 77.559 |
| Swap | G, B | 79.206 |
| Copy B | B, B | 79.035 |
| Copy G | G, G | 78.642 |

## Paired findings

- Same Base content moved into Gap instead of Base: +0.617 pp versus Base-only,
  SD of seed differences 0.572 pp, 5/5 seeds positive.
- Same Gap content moved into Base instead of Gap: -0.560 pp versus Gap-only,
  SD 0.287 pp, 0/5 seeds positive.
- Swap versus Normal: +0.024 pp, SD 0.275 pp, 2/5 seeds positive. **Essentially
  unchanged descriptively; do not claim an improvement or statistical equivalence.**
- Holding only the Gap slot: B performs +0.850 pp better than G, 5/5 seeds positive.
- Holding only the Base slot: G performs -0.793 pp worse than B, 1/5 seeds positive.

The classifier does not require a rigid original Base/Gap assignment to retain
aggregate performance: exchanging both contents is tolerated. However, the two
contents are not equally effective when used alone in the other slot. This is
**asymmetric functional substitutability**, not evidence that the representations
are identical, nor that slot identity is irrelevant in every condition.

The AV subgroup (Text missing) is especially vulnerable to placing only G in Base:
Normal 62.959%, Move G→Base 58.068%; Swap 63.265%. Corresponding Normal / Swap:
AT 87.073 / 86.974%, TV 86.848 / 86.840%. Full per-rate and per-seed tables are in
`move_swap_results/RESULT.md` and the CSVs; these subgroup summaries are not additive.

## Verification and provenance

- `run.py` only extends the offline readout patcher with three modes. Original
  memory scan generates Local/Base/Gap once per batch; no altered tensor feeds it.
- Targeted test: red (three missing modes), then green (1 passed), including
  donor removal, swap using original tensors, padding/ineligible rows and no mutation.
- All 40 normal replay evaluations match original logits exactly and reference masks.
- All five checkpoint hashes are identical to the preceding Copy experiment.
- Independent NPZ check: all prior five modes, labels and masks match the earlier
  saved outputs bit-for-bit in 40/40 artifacts. All new predictions are finite and
  ineligible rows remain identical to Normal.
- Old `results/` is preserved. New JSON/NPZ/CSV and tables: `move_swap_results/`.
- `summarize.py` accepts both historical five-mode and extended eight-mode artifacts.
- `git diff --check` passed. No production model/trainer files changed.

Existing `best.pt` files were historically selected using eight-rate-mean Test
W-F1. This audit fixes those checkpoints across all interventions and does not
claim access to per-rate-best weights. **INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL
PAPER RESULT.** No CKA or further intervention is automatically started.
