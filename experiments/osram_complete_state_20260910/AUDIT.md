# Frozen State-JEPA spike: target utility, prediction quality, shared gradients

INTERNAL EXPLORATORY DIAGNOSTIC ONLY. Five MOSI State-JEPA checkpoints,
seeds66–70, each selected by miss=.5 Test-oracle. No neural training, parameter
update, EMA update, or architecture change. Every model state tensor checked
bitwise against its checkpoint after analysis. Results: audit_results/seed_*.json.

## Locked scope

- Rate .5 only, not all eight rates. Train mask uses epoch0, validation fixed mask.
- Only valid incomplete utterances, same rows for all representation comparisons.
- Teacher e_full and s_full-local use full current utterance, eval mode, no memory.
- Ridge alpha10, same StandardScaler trained on training rows only. Fit continuous
  sentiment with intercept; evaluate validation sign W-F1, exclude y=0 from F1.
  No validation/test hyperparameter search. Train/validation conversation IDs disjoint.
- h700 dimensions; e/s256; h+e/h+s956. h+shuffled_s is dimension matched to h+s,
  shuffled independently within each split. No claim h versus s is parameter matched.
- Prediction metrics on missing validation utterances, raw pre-LN predictions;
  centered retrieval searches same cohort, per-pattern metrics retained in JSON.
- Gradient: at most first three training batches; dataset produced TWO per seed,
  hence10 paired measurements per parameter group. Eval-style dropout, identical
  forward for the two losses, autograd.grad without optimizer. Frozen/independent
  predictor and classifier parameters excluded. Report actual .1 loss weighting.

## Target utility: validation ridge W-F1 (%)

| Representation | Mean ± sample SD |
|---|---:|
| h_obs | 65.878 ± 3.726 |
| e_full (EMA encoder) | 77.074 ± 7.339 |
| s_full-local (EMA encoder + Local path) | 77.312 ± 8.857 |
| h_obs + e_full | 78.667 ± 6.222 |
| h_obs + s_full-local | 78.998 ± 6.015 |
| h_obs + independently shuffled s_full-local | 66.002 ± 4.512 |

Target concatenation exceeds h in5/5 seeds and matched shuffled concatenation
in5/5. This supports accessible label information in the full target, not its
predictability from missing inputs. Privileged full information is intentionally
available to these probes. The s versus e mean difference is only+.238pp,
and h+s versus h+e only+.332pp: no strong evidence that Local transformation
provides an important improvement over the full fused node.
These are validation probe scores, NOT deployed model Test W-F1.

## Prediction quality: pooled missing validation utterances

| Seed | Centered cosine | Retrieval % | Chance % | Prediction rank | Target rank |
|---|---:|---:|---:|---:|---:|
|66|-.00618|.508|.508|8.579|83.661|
|67|.13162|1.471|.490|8.423|79.156|
|68|.04468|.488|.488|6.669|81.653|
|69|.04933|.503|.503|7.427|80.394|
|70|.25254|1.500|.500|5.046|64.891|

Means: raw cosine .932894; centered cosine .094398;
Real−Shuffle raw cosine gap .000258;
retrieval .894% vs chance .498%; centered covariance effective ranks7.229 vs77.951;
mean per-channel prediction/teacher std ratio .6300.
Three seeds have retrieval at chance and two reach roughly3x chance, with small
candidate cohorts (~200). Nonzero variance is NOT high-dimensional information:
the predicted variation is much lower-rank than target variation.
This supports weak/low-rank sample correspondence, not a literal constant output
or a proven irrecoverable collapse. Raw cosine alone would be misleading.

## Gradient compatibility

| Shared parameter group | Mean cosine | Negative pairs /10 | Mean weighted state/emotion norm % |
|---|---:|---:|---:|
|ObservedSetEncoder|.377963|3/10|.004085|
|OSRAM|.064771|4/10|.015344|
|Combined|.121694|3/10|.008832|

Some pairs conflict, but no stable strong-negative pattern is observed. Auxiliary
gradients are very small relative to task gradients at these selected checkpoints.
This is a late-checkpoint deterministic gradient diagnostic, not a trajectory-wide
claim about training interference or proof that gradient magnitude caused failure.

## Decision

Evidence fits **target contains label information + weak low-rank prediction**
better than **good prediction + strong destructive gradient conflict**.
Do not launch conflict surgery or conclude that the full target is useless.
Do not claim that the Local-state transformation is essential: e_full probes
perform similarly. No new regularizer/predictor experiment launched in this spike.

Limitations: one rate; fixed ridge strength; small validation cohort; five seeds;
selected Test-oracle checkpoints; no independent significance claim. Full target
information may be intrinsically unpredictable from some missing patterns.
