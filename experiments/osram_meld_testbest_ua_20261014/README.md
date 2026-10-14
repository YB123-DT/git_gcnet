# Exact Test-best ACC and corresponding UA

INTERNAL DIAGNOSTIC ONLY. This recovers the actual masks used when each
retained checkpoint established its ACC-best score, then computes UA from
that same model/mask/label combination. UA is not independently maximized.

GCNet random-mask-v1 uses the global NumPy stream and ignores its schedule's
epoch argument. The historical trainer seeds that stream with
`SeedBundle(seed).derive('missing_m3_model_init:fold:5')`. Reproduce the exact
epoch-seeded training conversation order and padded batch rectangles, all
cyclic train mask draws, and all eight test-rate draws after each epoch.
Only masks are replayed; no model is trained. Current labels do not select
masks. Their array lengths define the existing padding rectangles.

Acceptance gates:

1. Replay through the final evaluation and exactly reproduce the retained
   final availability arrays at every rate.
2. For every selected epoch, inference must reproduce both checkpoint
   `selection_accuracy` and `selection_weighted_f1` within1e-10.
3. Only then report seven-class macro-recall UA from those predictions.

Inference exports selected predictions and masks to a new remote directory,
without overwriting training records or changing the missing sampler. Original
weights are still required: mask recovery cannot reconstruct deleted weights.

The earlier width `PROGRESS_UA.json` was removed from the current branch at
the user's request because it mixed primary selected ACC with secondary
mask-replay UA. Git history retains the old artifact, while current comparison
tables will use only the verified same-selection-mask ACC/UA pair.

One regression test checks draw order, host/guest selection and preservation
of global NumPy state. A real seed66 mask-only check already reproduced all
eight final masks exactly before this recovery script was implemented.
