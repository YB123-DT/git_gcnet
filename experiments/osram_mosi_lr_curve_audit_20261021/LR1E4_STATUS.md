# Clarifying the1e-4 segment and unusual early curves

Existing logs/code were inspected read-only; no training or inference added.

## What exists

The blue original Flat curve uses1e-3 for epochs1–100 and1e-4 for101–150.
That latter segment inherits100 epochs of1e-3 optimization and its Adam state.
It is NOT a run initialized and trained from scratch at1e-4, and must not be
labeled as an independent1e-4 comparator.

A targeted remote search of
`osram_*/execution*/runs/{MOSI,CMUMOSI}/seed_66/**/SUMMARY.json` found no
candidate with effective config learning_rate=1e-4. The earlier search of
`osram_mosi*/execution*/runs/**/SUMMARY.json` also found no seed66 original
Flat with this LR. This is a scoped search result, not a proof that every
historical layout or unindexed checkpoint lacks such a run. Old-mask or
different-model evidence cannot be silently substituted.

## Evidence for early optimization difficulty

Both original Flat schedules have the same initialization SHA256:
`89718ab6425657b9a05286892ff7f8d93bca0cac5891f3f0d5e601e49af26ef2`.
All150 training-mask, sample-count and update traces were already verified
equal. Both use gradient_clip_norm=1.0.

| Training epoch | Flat1e-3 MSE | Flat3e-4 MSE |
|---|---:|---:|
| 1 | 66.5468 | 20.7252 |
| 2 | 174.1757 | 20.8871 |

This supports a larger early loss excursion in the higherLR run, consistent
with overly aggressive early updates. It does not identify a faulty layer or
prove a unique causal mechanism. Recorded gradient_norm_max is PRE-clipping:
for example1376.6284 versus649.9931 at epoch1. Large recorded values do not
establish that clipping was absent or ineffective.

Three effects must remain distinct:

1. Only2 optimizer batches per epoch and cyclic8-rate training mean that an
   epoch is a small, changing-mask training window, not a fixed-condition
   loss estimate. Cross-run masks are paired, but within-run adjacent epochs
   do not have identical mask conditions.
2. Training optimizes continuous MSE; reported W-F1 filters neutral labels
   and thresholds predictions at zero (`train_gcnet.py::_metrics`). Crossing
   zero can change polarity classification abruptly without an equally abrupt
   change in regression error. No early-epoch prediction replay was performed
   to assert a specific collapse or bias mechanism.
3. PanelD is a cumulative per-rate maximum, mathematically nondecreasing by
   construction; its flat stretches do not show an idle optimizer.

No independent1e-4 curve is fabricated, and the request to inspect curves is
not treated as authorization for an additional training run.
