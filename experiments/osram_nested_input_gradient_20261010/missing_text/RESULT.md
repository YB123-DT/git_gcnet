# Flat versus original Nested: current Text-missing predictions

INTERNAL DIAGNOSTIC ONLY

Reuse the 48-condition cached predictions from the preceding input-gradient
diagnostic. No training, new inference, threshold search or weight change.
The original Flat and original Nested checkpoints were selected per rate under
the inherited Test-oracle protocol. This is a comparison of independently trained
full models, not a Gap-T on/off intervention.

## Cohort and aggregation

- Align seed, missing rate, conversation ID and utterance position; assert labels
  and current availability identical. All 16,464 original paired entries align.
- Select current patterns A/V/AV (Text unavailable). Histories retain the original
  random-missing masks: this is **not persistent whole-conversation Text missing**.
- There are 5,292 paired occurrences, of which 5,067 have nonzero labels.
  The same original utterance can appear under multiple seeds/rates; counts are
  paired occurrences, not unique independent utterances.
- W-F1 and accuracy: exclude gold label0 and classify prediction strictly `>0`.
  Continuous prediction shifts and MSE include all selected labels.
- Compute each seed/rate separately, then equally average nonempty seed/rate
  results. Rate0.0 has no Text-missing samples and is not assigned a dummy score.
  Overall is therefore 21 groups (3 seeds × 7 nonempty rates), not an eight-rate
  full-test-set mean. High missing uses nine groups (.5/.6/.7 × 3 seeds).
- Corrections/harms below are sums over those paired occurrences. Their difference
  need not equal macro W-F1 change; W-F1 is nonlinear and groups are equally weighted.

## Main results

| Cohort | Flat W-F1 (%) | Nested W-F1 (%) | Delta (pp) | Corrections | Harms |
|---|---:|---:|---:|---:|---:|
| All current no-Text | 67.376 | 66.872 | −0.503 | 440 | 475 |
| High missing .5/.6/.7 | 65.691 | 64.631 | −1.060 | 280 | 309 |
| All current no-Text, exclude first turns | 67.474 | 66.654 | −0.820 | 411 | 450 |

All current no-Text: mean absolute prediction difference is 0.301392, mean signed
Nested-minus-Flat change is +0.028325; macro polarity flip rate is 18.280%.
Of the 5,067 nonneutral paired occurrences, 2,916 are correct in both models and
1,236 are wrong in both. Flat MSE is 2.296145; Nested MSE is 2.349144.

## Current availability split

| Current pattern | Flat W-F1 (%) | Nested W-F1 (%) | Delta (pp) | Corrections | Harms | Mean absolute prediction change |
|---|---:|---:|---:|---:|---:|---:|
| A | 70.114 | 68.726 | −1.388 | 133 | 138 | 0.297292 |
| V | 64.235 | 61.684 | −2.551 | 173 | 203 | 0.341506 |
| AV | 66.694 | 66.387 | −0.307 | 134 | 134 | 0.288250 |

These subset scores are not comparable to separately imposed persistent patterns:
both the selected current samples and historical support differ. They also are
not a macro average of the three persistent no-Text patterns.

## Seed split, current no-Text

| Seed | Flat W-F1 (%) | Nested W-F1 (%) | Delta (pp) | Corrections | Harms | Mean absolute change | Flip (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 66 | 68.784 | 68.394 | −0.390 | 125 | 133 | 0.3060 | 16.462 |
| 67 | 67.453 | 68.043 | +0.590 | 135 | 130 | 0.2456 | 14.131 |
| 68 | 65.890 | 64.181 | −1.710 | 180 | 212 | 0.3525 | 24.247 |

## Rate split, mean over three seeds

| Rate | Flat W-F1 (%) | Nested W-F1 (%) | Delta (pp) | Corrections | Harms | Nonneutral occurrences |
|---|---:|---:|---:|---:|---:|---:|
| .1 | 70.138 | 68.972 | −1.166 | 21 | 22 | 199 |
| .2 | 69.322 | 71.170 | +1.848 | 48 | 42 | 416 |
| .3 | 68.166 | 67.794 | −0.372 | 38 | 39 | 576 |
| .4 | 66.931 | 66.278 | −0.653 | 53 | 63 | 745 |
| .5 | 66.420 | 65.671 | −0.749 | 69 | 76 | 911 |
| .6 | 65.736 | 64.930 | −0.807 | 92 | 96 | 1042 |
| .7 | 64.917 | 63.292 | −1.626 | 119 | 137 | 1178 |

## Interpretation limits

The smaller aggregate pre-Nested Gap-T input gradient from the preceding report
does not translate into an aggregate W-F1 improvement for these no-Text samples.
However, these checkpoints have different learned Local, Memory and task-head
parameters. This comparison does not prove that reducing Gap-T sensitivity caused
the performance difference, or that Gap-T itself is harmful. No causal mechanism,
statistical significance or training failure is claimed.

## Artifacts and verification

`missing_text_pairs.csv` contains all 5,292 paired predictions with seed/rate,
conversation/position, pattern, label, both predictions, signed change and outcome.
`missing_text_per_rate.csv` contains all seed/rate/pattern summaries;
`missing_text_macro.csv` and `SUMMARY.json` contain aggregation results.
No neural-network code is run by the comparison script.

Verified duplicate rejection, complete identity alignment, mask/label equality,
expected 3×8×686 source rows, and outcome-count partition for every reported group.
The binary W-F1 calculation retains the original evaluator's formula, nonzero-label
filter and strictly-positive threshold. Raw-source SHA256 is stored in SUMMARY.json.

Reproduction (output must be a new directory):

```bash
/home/yangbin/miniconda3/envs/msa_extract/bin/python \
  experiments/osram_nested_input_gradient_20261010/compare_missing_text.py \
  --input /tmp/osram-nested-input-gradient-20261010-utterances.csv \
  --output /tmp/osram-missing-text-comparison-replay
```

Original raw cache remains on biggpu at
`/data2/yb/remote_experiments/osram_nested_input_gradient_20261010/results/utterances.csv`.
