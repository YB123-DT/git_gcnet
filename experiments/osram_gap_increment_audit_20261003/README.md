# Same-checkpoint Base versus Gap audit

INTERNAL DIAGNOSTIC ONLY

Stage1 only, MOSI seed66, eight existing original cfg84 no-JEPA Flat per-rate BEST
checkpoints. Do not use separately trained local-base models or Relation checkpoints.
No training, new mask, second Memory trajectory, weight update or checkpoint selection.

The unchanged forward supplies Local, Base and masked Gap slots to the original
Flat adapter. Cache this input and Local Skip output, then replay only the adapter,
original emotion_norm and task head for three readouts:

- P_L: Base and all Gap slots zero; Local and Local Skip unchanged.
- P_B: Base retained, Gap slots zero; still a Local+Memory prediction.
- P_F: original complete Flat input.

Analyze P_B to P_F, not a comparison of separately trained models. For every valid
utterance retain squared-error difference (y-P_B)^2-(y-P_F)^2. Binary polarity uses
the existing label!=0 filter and prediction>0 threshold; neutral utterances remain
in raw records but do not enter rescue/harm labels or W-F1.

Norms and Base/Gap cosines use only the forward512 of1024 context dimensions.
Inactive Gap slots do not enter active-evidence means. Zero-norm cosine and
zero-Base relative norm are undefined rather than evidence of a meaningful zero.
Existing _scan residual-address diagnostics are collected without changing formulas;
rho/eta/query cosine are summarized over heads, then over currently active Gap slots.

Report observable distributions and signed raw AUROC (rescue=positive) separately
by rate and by current availability. An AUROC below0.5 is inverse association,
not absence of signal. Do not fit thresholds, train a Gate or choose routing using
test labels. Different rate checkpoints and repeated utterances preclude treating
all rate exposures as independent validation examples. Tiny rescue/harm strata
must show their class counts; undefined single-class AUROC must stay undefined.

Replay validation requires exact labels/availability, original sample IDs and
Full polarity, with FP32 prediction absolute tolerance1e-6. Model tensors and
checkpoint file hashes must remain unchanged. At fully observed positions P_B
must equal P_F; at first valid positions P_L=P_B=P_F because there is no history.

Head ablation and query-specialization analysis are conditional later stages,
not automatically justified by these exploratory associations.
