# MOSI Text-Anchor Residual Fusion Design

## Objective

Test one causal response to the five-seed fusion localization result: when
Text is observed, preserve the learned T-only Slot node and allow Audio and
Visual to enter only through a bounded residual. This is a screening variant,
not yet a novelty claim.

## Alternatives considered

1. **Hard Text-only anchor:** discard Audio/Visual whenever Text exists. This
   preserves Text but cannot test whether the measured correspondence signal
   is usable.
2. **Text-anchor bounded residual (selected):** reuse the existing Slot encoder
   to compute both the normal observed-set node and the corresponding T-only
   node; learn a small residual from their difference.
3. **Learned anchor selection:** more general, but introduces routing and makes
   the first causal test harder to attribute.

## Exact behavior

Let `f_slot(S)` be the existing Slot fusion and `f_slot(T)` the same weights
evaluated with only the Text slot and T pattern embedding.

- `T`: output exactly `f_slot(T)`.
- `AT`, `TV`, `ATV`:

  `node = f_slot(T) + Bound_k(MLP([f_slot(T), f_slot(S)-f_slot(T), product]))`

- `A`, `V`, `AV`: output the unchanged `f_slot(S)` fallback.
- padding remains zero and missing raw blocks remain unread.

`Bound_k` limits each residual norm to at most `k=0.25` times the Text-anchor
norm. The residual output layer is zero initialized. New-module construction
uses an RNG fork so common downstream parameter initialization remains paired
with the Slot control.

## Locked experiment

- CMU-MOSI, seeds 66--70, 100 epochs.
- Existing cyclic eight-rate training and eight-rate validation selection.
- Same frozen wav2vec/DeBERTa/MANet features.
- Same GCNet, Missing-M3 predictor, JEPA loss, optimizer, masks, and readout.
- Original Slot five-seed results are inherited and not rerun.
- Primary comparisons: eight-rate mean, miss=0, miss=0.5, miss=0.7, and
  per-seed paired deltas.

## Falsification

The screen fails if the five-seed eight-rate mean does not improve, if miss=0
falls materially, or if gains arise only from one seed. Failure closes this
hard Text-anchor implementation; it does not invalidate the localization
diagnostic.

