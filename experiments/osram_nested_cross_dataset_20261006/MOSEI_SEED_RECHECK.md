# MOSEI seed and BEST timing recheck

INTERNAL DIAGNOSTIC ONLY. Existing 100-epoch per-rate Test-oracle BEST, eight-rate means; no new training/inference. Sources: REQUESTED_METRICS_20261008.json and best_epochs/RESULT.md (exact epochs and provenance in best_epochs/SUMMARY.json).

| Seed | Flat ACC | Nested ACC | Flat W-F1 | Nested W-F1 | Delta W-F1 (pp) | Flat BEST epoch range | Nested BEST epoch range |
|---|---:|---:|---:|---:|---:|---:|---:|
|66|83.799|83.988|83.671|83.881|+0.210|8–29|20–29|
|67|84.005|84.088|83.886|83.877|−0.010|13–39|19–36|
|68|83.998|83.909|83.836|83.772|−0.064|7–39|22–34|
|Mean|83.934|83.995|83.798|83.843|+0.046|—|—|

W-F1 gain comes from seed66; seed67 is nearly unchanged and seed68 decreases. ACC improves for seeds66/67, decreases for68. All selected checkpoints occur by epoch39; specifically Nested67's final selected set is attained by epoch36. These are per-rate selected epochs, not one common best checkpoint. This does not prove a changed schedule could not help. The recent 150-epoch completion batch did not include MOSEI.
