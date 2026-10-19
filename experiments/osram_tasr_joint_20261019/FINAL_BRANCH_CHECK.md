# Completed saved-artifact branch check

INTERNAL TEST-ORACLE SCREEN. Verified 2026-10-19 after SSH recovered.
Read existing SUMMARY.json, selected prediction NPZ files, previously completed
paired replay NPZ files and best_miss_0p3.pt on biggpu. CPU artifact analysis only:
no training, new model inference, mask resampling, weight changes or GPU job.
Paths and training source c5c26dd are recorded in RESULT.md.

## Selected predictions versus matched original Flat

All eight selected pairs have exactly equal labels, availability, conversation
IDs, utterance indices and speakers. Each contains 2610 utterances.

| Rate | Different predicted classes | Maximum absolute logit difference |
|---|---:|---:|
|0.0|0|0.00021720677614212036|
|0.1|0|0.0005958080291748047|
|0.2|0|0.0004693269729614258|
|0.3|0|0.0006704330444335938|
|0.4|0|0.00124359130859375|
|0.5|0|0.0005792379379272461|
|0.6|0|0.0007991790771484375|
|0.7|0|0.0021217763423919678|

Thus equal ACC/UA/W-F1 is confirmed at sample level, not merely an aggregate
coincidence. Logits are NOT identical, including at rate0 where TASR Gap correction
is inactive: joint training can change original parameters through other batches.

## Same joint checkpoint: saved TASR-on versus TASR-off replay

This is a separate comparison: off retains the jointly trained Flat/Memory/head,
not the original separately trained Flat. All 15 conditions have 2610 samples.
There are zero corrections and zero harms in every condition. The one class
change at random0.7 is wrong-to-another-wrong, not a correction or harm.

| Condition | Different classes | Maximum absolute logit difference | Mean recorded delta/original Gap norm ratio |
|---|---:|---:|---:|
|Random0.0|0|0|0|
|Random0.1|0|0.000804901123046875|0.0007757225927586356|
|Random0.2|0|0.000492095947265625|0.0005778595611142615|
|Random0.3|0|0.0009632110595703125|0.0008873807253419525|
|Random0.4|0|0.0022585391998291016|0.0018294936097744438|
|Random0.5|0|0.0008141994476318359|0.0007026232132274243|
|Random0.6|0|0.0013804435729980469|0.0010481561896287734|
|Random0.7|1|0.004128932952880859|0.003279542705665032|
|Persistent A|0|0.0002846270799636841|0.0007219770535205802|
|Persistent T|0|0.0006347596645355225|0.0006125993548064596|
|Persistent V|0|0.0002416372299194336|0.0004642772805204408|
|Persistent AT|0|0.0004649162292480469|0.0006399625110336476|
|Persistent AV|0|0.00036215782165527344|0.00044240732030529116|
|Persistent TV|0|0.0008150339126586914|0.0007498632055810756|
|Persistent ATV|0|0|0|

The norm-ratio column is the unweighted arithmetic mean of saved batch diagnostic
values, not a newly recomputed sample-weighted statistic or maximum. Expressed
as percentages: approximately0.058--0.328% for random rates with missingness and
0.044--0.075% for the six Persistent missing patterns.

## Learned parameters and gradient connection

- In the selected miss0.3 checkpoint, all64 output score weights are nonzero;
  weight norm0.00595603883266449, scalar bias0.00020150459022261202.
- Last saved gradient audit: finite=True; TASR12/12 parameter tensors connected
  and nonzero. Encoder30/30, Flat6/6, classifier2/2, Query2/2, writer key6/6,
  writer value6/6 also connected and nonzero in that audit.
- These are a selected checkpoint weight check and last-step gradient audit,
  not a claim that every tensor had a nonzero gradient at every training step.

## Bounded conclusion

The trained branch is connected, updated and numerically influences logits;
there is no evidence here of an entirely bypassed/dead branch. Its observed
Gap correction is small and produces no accuracy-changing rescues or harms in
the saved replay. The one-stage run provides no classification benefit versus
the matched original Flat. This does not prove that all trajectory/source
methods fail, that zero initialization is the cause, or that Memory lacks signal.

Stop automatic LR/seed expansion of this configuration. Preserve implementation,
checkpoints and negative results; no new method or training is authorized by
this artifact check. The limited IEMOCAP6/fold5 frozen result remains separate
and does not establish a broadly effective second Block.
