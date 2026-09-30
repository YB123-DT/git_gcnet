# Frozen readout-path diagnostic — MOSI completed

## Scope and verification

3 seeds ×8 rates ×2 splits ×13 settings =624setting cells,48 prediction files.
All completed; each identity replay bitwise matches original logits, model-state
hashes unchanged, archived prediction file hashes independently rechecked.
No training, new Gate, test inference or deployable coefficient selection.
Train52conversations/1284utterances; validation10independent conversations/229.
Checkpoint origin remains historical per-rate Test-oracle; not formal results.
IEMOCAP not run because existing official validation aliases test. No claim of
cross-dataset replication or IEMOCAP CE result.

## All13 predeclared settings

Delta relative to(1,1,1). MSE lower is better; W-F1 differences in percentage
points. Rate macro within seed then seed macro. Transition counts sum24seed/rate
evaluations, not unique utterances; nonzero labels and >0 threshold unchanged.

|alpha|beta|mu|Train ΔMSE|Train ΔW-F1 pp|Validation ΔMSE|Validation ΔW-F1 pp|Wrong→right|Right→wrong|Val seeds MSE↓/F1↑|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0.8|0.8|0.8|+0.018787|-0.090|+0.014465|+0.327|49|32|0/3; 2/3|
|0.8|0.8|1|+0.038200|-0.227|+0.035070|+0.441|77|55|0/3; 2/3|
|0.8|0.8|1.2|+0.057908|-0.497|+0.056435|+0.025|91|91|0/3; 2/3|
|0.8|1|1|+0.012622|-0.081|+0.013145|+0.028|23|22|0/3; 2/3|
|1|0.8|1|+0.018787|-0.090|+0.014466|+0.327|49|32|0/3; 2/3|
|1|1|0.8|-0.007613|+0.028|-0.007932|-0.024|26|26|3/3; 1/3|
|1|1|1|0.000000|0.000|0.000000|0.000|0|0|0/3; 0/3|
|1|1|1.2|+0.009954|-0.031|+0.010350|+0.103|19|14|0/3; 2/3|
|1|1.2|1|-0.008747|+0.023|-0.005292|-0.055|26|28|2/3; 1/3|
|1.2|1|1|-0.006596|-0.001|-0.006860|-0.135|19|25|3/3; 1/3|
|1.2|1.2|0.8|-0.012286|-0.077|-0.009261|-0.469|70|90|3/3; 0/3|
|1.2|1.2|1|-0.011996|+0.007|-0.008664|-0.155|45|51|3/3; 2/3|
|1.2|1.2|1.2|-0.008747|+0.023|-0.005292|-0.055|26|28|2/3; 1/3|

## What this does and does not establish

- Reducing Memory alone(mu=.8) and increasing adapter Local alone(alpha=1.2)
  reduce validation MSE in all3seeds, but mean W-F1 changes are−.024 and−.135pp.
  Both improve conversation-averaged MSE on6/10validation conversations, and
  their descriptive conversation bootstrap intervals cross0. This is a directional
  MSE response, not demonstrated reproducible classification improvement.
- Reducing Local Skip alone(beta=.8) yields validation W-F1+.327pp with49
  corrections/32 harms, but MSE+.014466 (worse in all3seeds). W-F1 seed deltas:
  −.259,+.369,+.872pp. Train W-F1−.090pp. Not a consistent across-seed/split win.
- Increasing Local Skip alone(beta=1.2) yields MSE−.005292 but W-F1−.055pp;
  MSE improves2/3seeds and6/10conversations. Again no clean joint improvement.
- No non-identity setting improves validation W-F1 in all3seeds. The13-way
  diagnostic does not currently establish a direct path adjustment with stable
  task-loss AND sentiment-classification benefit. It does not prove adaptive
  gates impossible or identify training failure causally.

## Important LayerNorm interpretation

Original emotion_adapter begins with LayerNorm over[Local,C] (osram.py).
Uniform positive scaling of its entire input is approximately canceled by that
normalization, up to epsilon/numerical effects. Consequently(.8,.8,.8) has the
same classification predictions here as(1,.8,1), and(1.2,1.2,1.2) as(1,1.2,1).
Verified across all48artifacts: zero sign differences for both pairs; maximum
absolute regression-logit differences8.2254e-6 and4.0531e-6 respectively.
All13settings remain reported; these near-redundant responses must not count as
independent confirmation of a Local-versus-Memory mechanism. Adapter-only versus
Memory-only controls mainly alter their relative input magnitudes.

## Detailed evidence

- analysis/by_seed_rate_pattern.csv: allseed/rate/setting/availability losses,
  metrics, corrections and harms. Availability subsets are random-missing
  utterances, not persistent missing-modality trajectories.
- analysis/by_conversation.csv: individual-conversation paired responses.
- analysis/SUMMARY.json: seed means/SD, pattern summaries, conversation bootstrap.
- results/seed_*/: all13predictions, IDs, availability, provenance and state hashes.

Bootstrap uses2000conversation resamples after averaging repeated seed/rate
measurements within each conversation. Intervals are descriptive, unadjusted for
multiple comparisons and based on only10validation conversations. No significance
or deployable coefficient selection claim.

NEXT ACTION (not executed): inspect already saved per-conversation/per-availability
cases for why MSE-improving and sign-correcting changes disagree, before selecting
another Gate structure. No new training or inference is required for that check.
