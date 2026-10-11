# Frozen Base–Gap output interaction

INTERNAL DIAGNOSTIC ONLY. MOSI seed66, eight original per-rate Test-oracle
BEST checkpoints from osram_nested_training_gradients_20261010/runs/flat_seed66.
No training, no checkpoint selection, no mask change. One complete causal scan
per batch; replay the same Local/Memory through original Flat/norm/task head.

Compute f00, f10, f01, f11 and I_BG=f11-f10-f01+f00. Compare full prediction
with the synthetic additive output f10+f01-f00. Report per-rate W-F1, macro
eight/high rates, corrections/harms, interaction distributions, current
availability and exact adjacent nonneutral same/opposite polarity groups.
The latter labels are analysis-only, never model inputs or routing rules.

Check frozen hashes, original labels/masks/predictions, replay parity, inactive
Gap and padding. No-active-Gap must have zero interaction within numerical
tolerance. Masking observed reads may be OOD; nonzero interaction does not
establish that interaction is necessary, semantic conflict, or emotion shift.
