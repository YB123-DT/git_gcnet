# Current Missing-M3 JEPA Contribution Ablation

## Question

Does the training-only Missing-M3 JEPA objective improve the current Slot +
all-rates model beyond the observed-set construction and mixed-rate exposure?

## Locked comparison

The treatment is the completed current model with `jepa_weight=0.1`.  The new
control changes only `jepa_weight` to `0.0`; model construction, forward calls,
EMA lifecycle, RNG consumption, masks, optimization, validation selection, and
test evaluation remain unchanged.

Run IEMOCAP-6, fold 5, seeds 66--70 first.  Use one checkpoint per seed selected
by the mean validation W-F1 across missing rates 0.0--0.7, then evaluate that
checkpoint on all eight fixed test masks.  Existing treatment results are
inherited and must not be retrained.

## Decision gate

JEPA contributes on IEMOCAP-6 only if both the eight-rate and high-missing
paired mean deltas (`with JEPA - no JEPA`) are positive and at least three of
five seeds are positive for each aggregate.  Only after that gate passes may
the same control be run on CMU-MOSEI.  CMU-MOSI already has this ablation;
IEMOCAP-4 is not part of this gate.

The result supports only a training-time representation-regularization claim.
It does not establish test-time modality completion.

