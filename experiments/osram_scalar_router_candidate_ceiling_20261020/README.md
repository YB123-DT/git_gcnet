# Existing Joint Router candidate ceiling

INTERNAL POST-HOC ORACLE DIAGNOSTIC ONLY

Question: can the existing three scalar candidates provide enough polarity
alternatives for a better Router, or do they mostly remain on the same side
of the existing zero threshold? This audit adds no model or learned rule.

Primary completed runs:

- MOSI main seeds66/67/68:150epochs, original backbone1e-3/1e-4 schedule.
  Remote`/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_{seed}/joint`.
- MOSEI main seed66:60epochs, original backbone1e-3 constant.
  Remote`/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/joint`.
- Supplement, kept separate: completed MOSI seed66 constant1e-5.
  Remote`/data2/yb/remote_experiments/osram_mosi_router_lr1e5_20261020/execution_v1/runs/MOSI/seed_66/joint`.

All use GCNet random missing and the originally selected per-rate Test-best
W-F1 weights. The currently running2e-5 experiment is excluded: no partial
run is selected or evaluated for this diagnostic. No new training or inference.

Read`SUMMARY.json`,`best_by_rate.json` and saved`evaluation/random_0pX.npz`.
Saved candidate order is `[Full, remove Gap-T←T, remove Base←T]`, confirmed
against`JointTextPathModel.forward` and`RegressionTextPathRouter.forward`.
The comparison Full is from the SAME Joint checkpoint, not separately trained
Flat. It still contains Base/Gap. This distinction isolates routing from the
jointly trained backbone's drift.

Reproduction (CPU only; repeat`--case` for each completed run):

```bash
python -m experiments.osram_scalar_router_candidate_ceiling_20261020.analysis \
  --case MOSI_main_seed66=/path/to/saved/seed_66/joint \
  --case MOSEI_main_seed66=/path/to/saved/CMUMOSEI/seed_66/joint \
  --output /new/independent/analysis_directory
```

Validation checks selected NPZ SHA256 against both ledgers, selected epochs,
same-call mask SHA256, exact utterance/label order across rates and all four
saved scalar metrics. Eligibility is reconstructed from strictly past Text
availability, not trusted blindly. Ineligible candidates must be identical.
Weights must be nonnegative and normalized; saved predictions must reproduce
their convex mixture. No logits or masks are regenerated.

Counts and interpretation:

- Classification uses original`gold!=0` filtering and`prediction>0`.
- Same-polarity percentage uses eligible nonzero rows as denominator, not
  all rows. Empty cohorts such as rate0.0 eligible are N/A.
- Correctable: Joint Full is wrong and at least one candidate is correct.
  Actual Router corrections/harms use this same checkpoint and these same rows.
- Class oracle preserves Full-correct rows and repairs only correctable rows,
  using the correct candidate with lowest MSE. Labels are used only AFTER
  evaluation. It is not deployable and not a proven global W-F1 maximum.
- Separate MSE floors use the closest discrete candidate or projection of the
  label onto their scalar convex hull. These use labels and are bounds only.
- MSE includes neutral gold0, unlike polarity metrics; conditional outcome
  tables exclude neutral rows. Delta MSE always means Router minus Full.
- Mean/high metrics first average the original8/3rates separately per seed.
  Pooled counts are repeated utterance-rate occurrences, not unique samples.
  No claim of independent significance is made from repeated rates.
- Nearly unanimous candidates constrain polarity routing but do not imply
  that Memory is useless; a convex mixture may still change regression MSE.

Outputs: RESULT.md,PER_RATE.csv,MSE_GROUPS.csv,SUMMARY.json. Preserve original
weights, predictions, masks and selected-epoch records unchanged. Large NPZs
are not committed. The source contains no Torch import or model forward.
