# Unified CAM ablations, 2026-10-30

Scope approved in chat: organize original Flat/CAM checkpoint inference and the
separate frozen Nested second stage; then run one seed of A0–A3 from scratch.
Dataset was not explicitly named in the final approval. The assistant announced
MOSEI seed66, 60 epochs, constant LR1e-4 as a bounded continuation of the current
MOSEI work. No other dataset, seed, HPO, or second-stage training is launched.

## Implementation order

1. Audit actual current HEAD, accepted parents, exact masks, environment, and
   checkpoint format. Do not substitute old s0 `setup()` functions.
2. Keep native model/sampler files unchanged. Use the existing task-input
   ablation switches; retain the same architecture and initial tensors.
3. Provide one folder of configuration, training, inference, checkpoint loading,
   scheduler, tests, and reports. Native model packages are included in the
   immutable execution snapshot; they are not reimplemented.
4. Test precise slot masking, raw/residual query behavior, gradient reachability,
   strict checkpoint restoration and scheduler admission. Run one real-batch
   GPU check before training, not a sweep of smoke experiments.
5. Freeze source/config/data/mask identities, launch exactly four independent
   jobs on healthy available biggpu devices. Never use physical GPU4, alter
   other tasks, or change batch32 to accommodate concurrency.
6. Publish exact selected-epoch predictions and mask identity, eight-rate and
   high-missing summaries. Preserve failures and distinguish running/completed.

## Four task paths

| ID | Emotion input | Missing-info address |
|---|---|---|
| A0 | Local, zero Context, zero Missing-info | Unused by task |
| A1 | Local + Context, zero Missing-info | Unused by task |
| A2 | Local + Context + Missing-info | Raw query |
| A3 | Local + Context + Missing-info | Original residualized query |

Context = original Base; Missing-info = original three Gap slots. Local Skip
is preserved in every group. All groups retain the 4352-dimensional adapter
input, width1600 and the original parameter tree. A0/A1 are capacity-matched
task-path ablations, **not** reduced-parameter architectures: CHM may still scan
for matched computation/diagnostics but its suppressed outputs cannot enter the
emotion loss. Gap must still be zero for currently observed modalities.

The joint A/T/V write, decay/read/write order, projector definitions, eta0.6,
causal zero backward half, original loss, original GCNet random sampler and
training data remain unchanged. No auxiliary objective or additional block.

## Protocol and interpretation

MOSEI66, official split, batch32, Adam1e-4, WD1e-5, dropout as accepted config,
60 epochs, cyclic rates0.0–0.7. Use accepted fixed GCNet test-mask bank for all
arms; validate ordered sample identity, never generate fresh evaluation masks.
Per-rate TEST W-F1 selection; ACC and MSE come from those same predictions.
Label every result INTERNAL TEST-ORACLE SCREEN. An eight-rate mean may combine
different checkpoints; it is not a single-checkpoint deployment result.

Freeze original parent and train only original Nested parameters in any future
second-stage run. Such runs require separate authorization and an explicit
own-rate parent checkpoint. Block-only checkpoints must never be mistaken for
complete original Flat weights.
