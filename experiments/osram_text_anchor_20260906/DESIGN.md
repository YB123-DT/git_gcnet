# OSRAM + Text-anchor residual

## Scope

This is a single-variable diagnostic against the completed `OSRAM + mean`
run. The conversational backbone, Student/EMA/JEPA predictor, optimizer,
cyclic mixed-rate schedule, masks, features, fold, and seeds are unchanged.

The only changed model setting is:

```text
fusion_type: mean -> text-anchor-residual
```

The variant uses the existing `ObservedSetEncoder` implementation. It first
uses the slot-conditioned fusion path and, when Text is observed together
with Audio or Visual, computes a Text-only anchor and applies a zero-initialized,
rank-64, norm-bounded residual. When Text is missing (or is the only observed
modality), the anchor residual is inactive; the rest of the OSRAM path is
unchanged.

## Hypothesis

Text is the strongest and most stable MOSI modality. Anchoring an observed
Audio/Text or Text/Visual node to a Text-only representation may reduce
destructive cross-modal averaging while retaining an explicitly bounded
correction from the additional observed modality.

## Protocol

- Dataset: CMU-MOSI, regression task, fold 1.
- Missing rates: 0.0 through 0.7, cyclic training schedule.
- Seeds: 66, 67, 68, 69, 70.
- Checkpoint selection: one epoch per seed by the mean Test weighted-F1 over
  all eight rates (`8-rate-mean-test-oracle`).
- Additional diagnostic: per-rate Test-oracle maxima, reported separately;
  this selects a different epoch for each rate and is not a formal result.
- Features: wav2vec-large-c-UTT, deberta-large-4-UTT, manet_UTT.
- All other settings match `experiments/osram_complete_20260906/raw/*/config.json`.

This directory is an internal diagnostic only. It must not be presented as a
validation-selected paper result.
