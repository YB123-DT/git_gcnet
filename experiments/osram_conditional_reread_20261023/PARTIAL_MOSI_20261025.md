# MOSI conditional reread — interrupted results verified 2026-10-25

INTERNAL TEST-ORACLE SCREEN — INTERRUPTED/PARTIAL, NOT COMPLETED150.

No training was resumed, no fresh inference ran, and no existing result or checkpoint was overwritten.
Remote process checks show both MOSI workers absent. Stale running labels in their original
SUMMARY.json were not rewritten; CANCELLED_20261023.json records the intentional stop.

## Retained Test-best scores

| Model | Observed training progress | Mean8 ACC | Mean8 W-F1 | High W-F1 | Mean8 MSE |
|---|---|---:|---:|---:|---:|
|Flat|150/150|79.268293|79.249087|73.409792|1.494741|
|free|109/150|79.192073|79.038291|72.920965|1.467595|
|span|108/150|79.077744|79.038533|73.138785|1.490052|

Free minus span mean8 W-F1: -0.000242pp.
Free minus Flat mean8 W-F1: -0.210796pp.
Span minus Flat mean8 W-F1: -0.210553pp.

## Per-rate W-F1 and selected epoch

| Rate | Flat W-F1 | Flat epoch | Free W-F1 | Free epoch | Span W-F1 | Span epoch |
|---|---:|---:|---:|---:|---:|---:|
|0.0|87.250577|48|87.487354|59|87.584151|41|
|0.1|83.871212|52|83.968342|66|84.676934|44|
|0.2|82.595054|49|83.146474|73|82.308325|44|
|0.3|80.741193|50|80.558697|79|79.617422|47|
|0.4|79.305280|62|78.382565|37|78.705077|44|
|0.5|75.995830|48|75.223930|92|76.009877|61|
|0.6|75.327887|48|74.491681|37|73.152691|47|
|0.7|68.905659|51|69.047284|98|70.253787|47|

Both interrupted variants' current eight BEST checkpoints were selected before epoch101:
Free maximum selected epoch98, Span maximum66. This does not prove that later epochs
could not improve or establish equivalence after a completed matched150-epoch budget.

## Protocol and verification

- Seed66, cfg84 no-JEPA LargeFlat1600, original GCNet random mask v1/cyclic0–0.7.
- Original training environment is s0 (Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2),
  NOT the more recent cmerc Flat/Nested screen. Only independent prediction statistics
  were recomputed in cmerc-repro; no model was rerun or environment switched for training.
- MOSI training budget150, LR3e-4 through epoch100 then3e-5, batch32/Adam/WD1e-5/MSE.
- Free109 and Span108 are interrupted; Flat150 is complete. Do not present their
  current scores as an equal-completed-budget comparison or independent generalization.
- All24 retained prediction SHA256 checked against original per-rate manifests.
  ACC/W-F1/MSE independently recomputed from exact saved predictions with NumPy/sklearn.
- Exact labels, availability masks, conversation IDs and utterance indices match among
  all three models at every rate. Flat SUMMARY SHA256 matches the recorded reference.
- No model/prediction/mask was modified. No automatic resume, new epoch selection,
  additional seeds or hyperparameter search was performed.
- Correct criterion:Test-best per-rate W-F1, with ACC/MSE from those same predictions.
  Mean8 may combine different epochs and is not one deployed checkpoint.

## Bounded interpretation

Current free/span W-F1 differs by less than0.001pp and neither exceeds the completed
matched-protocol Flat reference. Free has lower MSE, but that is not a W-F1 gain.
These partial scores provide no positive support so far for improved task prediction
from accessing query directions outside the original active-query span.
This is not a completed null result, statistical equivalence proof or evidence
that all second-query mechanisms must fail.

Original implementation source:`8f15f5315e1bd23fa89765d4a8344128719858e7`.
Original outputs:`/data2/yb/remote_experiments/osram_conditional_reread_20261023/execution_v1`.
Exact per-rate scores, source prediction/checkpoint hash references and environments
are preserved in PARTIAL_MOSI_20261025.json.
