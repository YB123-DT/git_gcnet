# MOSEI near-uniform V3 output attention: evidence and limits

INTERNAL TEST-ORACLE CHECKPOINT DIAGNOSTIC

This note reinterprets existing artifacts only. No new training, inference,
checkpoint modification or attention intervention was performed.

## Verified observations

The completed standard V3 seed66 audit replayed eight saved rate-specific BEST
checkpoints with their original exact test masks. In the focus group (current
Text missing and all historical A/T/V sources seen), there are 7,737 pooled test
slots, not 7,737 independent unique utterances or multiple seeds.

- Base and Gap-T source-weight means both round to [0.3333, 0.3333, 0.3334].
- Reported entropy is computed for each utterance, including each individual
  attention head, before averaging. Thus this is not merely cancellation of
  selective heads or different samples' source preferences.
- The pooled normalized entropy is above 0.9999998. Individual rate/role rows
  have small departures (e.g. rate0.4 Gap-T per-head entropy about 0.99999836).
  Values infinitesimally above one in CSV are floating-point rounding, not
  super-uniform distributions.
- Mean Base-versus-Gap-T weight L1 distance is 0.000056.
- Matched other-conversation Local-query swaps change mean weight L1 by
  0.000020 for Base and 0.000010 for Gap-T, retaining original keys/values.

Sources: RESULT.md, VALIDATION.md, FOCUS.csv, HEADS.csv and stats.py::_row.

## What follows, and what does not

For attention logits l_s, softmax(l) is uniform when the logits are equal;
near-uniform output indicates weak relative logit separation at this final
cross-attention stage. For one head with three active sources, the output
therefore approximates W_O[(v_A+v_T+v_V)/3], accounting for the usual value and
output projections. It need not be zero.

Possible explanations include weak projected queries, nearly identical
projected source keys, or optimization that does not favor discriminative
query-key matching. These are hypotheses, not causes established by the saved
attention statistics. Layer2 already mixes source tokens; equal output weights
also do not imply equal causal contributions of original A/T/V observations.
Local additionally enters the pair encoder, so a weak final query effect does
not prove all Local conditioning is absent.

For the intended adaptive source-selection mechanism, these observations are
unfavorable: the final selector exhibits little source or query differentiation
in this cohort. They do not establish that uniform pooling is intrinsically
bad, that the complete branch is inactive, or that it causes the near-unchanged
aggregate W-F1. A learned uniform aggregate can still carry useful content;
similar aggregate scores can hide per-sample corrections and harms.

Do not force sharp attention or lower its temperature merely to produce more
visually selective weights. Selectivity is not a task-performance objective.
The existing evidence does not validate a successful adaptive-selection story.
If directly testing that claim later is authorized, a same-checkpoint learned-
versus-uniform final-attention intervention is narrower than retraining a new
selector, but has not been performed or claimed here.
