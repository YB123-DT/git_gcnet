# Current-missing original Nested: preflight evidence

INTERNAL TEST-ORACLE SCREEN. Tests and smoke cannot establish effectiveness.

## Reused mechanism

The canonical entry is `nested_gnn_rooted_evidence`, verified in actual factory
code and the historical MOSEI cross-dataset record. Existing dim64/depth3/hidden64
NestedGNN, eight Memory-head groups, rooted one-hop subgraphs and zero-initialized
input decoders are reused unchanged. Original Flat1600 and original Local Skip
remain. No model source, Memory formula, mask sampler or task-loss change.

Old Nested three-seed W-F1 was83.843445, but the historical33ded09 trainer used
ConversationMaskSchedule, so that number cannot be a current GCNet control.
The tuned dim32/depth2 candidate is a different configuration, not silently used.

## Scope and controls

Five new jobs only: Nested66/67/68, Flat67/68; reuse completed accepted Flat66.
All current protocol runs: MOSEI official splits,60 epochs, constant1e-3, Adam,
batch32, WD1e-5, s0, no-JEPA, original task MSE, cyclic0.0..0.7 GCNet random masks,
per-rate TEST-best W-F1. Exact selected predictions/masks, eight BEST and full
last model/optimizer/RNG are retained. No learning-rate or architecture search.

Direct67/68 runs are used ONLY to verify data, initialization, random seeds,
training-order/mask traces and evaluation masks. Their performance is never
reported as Flat. Existing low-LR Flat67/68 do not supply same-LR controls.

## Tests-first and review

Fourteen worker tests initially failed for the missing experiment module; after
implementation they passed. Queue/report tests also followed a red/green cycle.
After increasing the resource floor (a first-three-batch smoke is not a guarantee
of worst-case0.7 mask memory), the failing resource-floor test was fixed by
enforcing at least8GiB per Nested task. Flat reserves4GiB, plus2GiB card margin.

Local worker/queue/report tests:30 passed. Remote s0 worker/queue/report and
existing Nested regression tests:38 passed in9.18s. Python compilation and diff
checks passed. Measured original Nested addition159,235 parameters; unchanged
original Flat13,509,793 gives13,669,028 total. Real GPU smoke remains a separate
pre-training gate.

The worker checks original parameter digest and zero-initialized prediction
parity, exact config/data/split/seed/mask reference, one causal scan per update,
and nonzero tokenizer/GIN/pool/readout/decoder task gradients after three smoke
steps. Normal task updates/clipping/optimizer are delegated to the current
shared training loop; original sampling is not rewritten. Full-state restore
and exact selected replay are also checked by the GPU smoke.

Independent read-only review of the worker found no blocking protocol issue.
Report tests reject mismatched actual Flat/Nested LR, data, split, mask,
initialization and seed protocol; a successful worker flag alone is not accepted
as proof of a final pair. Incomplete results cannot enter three-seed averages.

No positive task result is claimed at this preflight stage. Actual immutable SHA,
GPU cost, paths and process IDs must come from subsequent execution evidence.
