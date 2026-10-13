# IEMOCAP Correct-Mask Nested Completion Design

## Goal

Complete the missing matched Nested GNN comparison for IEMOCAP-4 and
IEMOCAP-6 under the original GCNet batch-level `random_mask` protocol.

## Fixed comparison

The completed correct-mask Flat runs are the anchors. The new runs copy their
dataset, feature roots, seeds 66/67/68, five held-out folds, 100-epoch budget,
optimizer, learning rate, batch size, task loss, causal OSRAM, Flat readout,
and per-rate Test-oracle selection. The sole model delta is:

```text
osram_meaningful_block:
  none -> nested_gnn_rooted_evidence
```

No parameter sweep, new loss, persistent missing schedule, or checkpoint
continuation is part of this completion run.

## Execution

IEMOCAP-4 is assigned to healthy physical GPU0 and IEMOCAP-6 to healthy
physical GPU1 on `biggpu`. Each card admits at most three fold jobs at once.
Every run writes an isolated config, provenance, history, per-rate metrics and
predictions. Disposable weight files are removed only after a successful run.

## Acceptance criteria

- Exactly 30 jobs: 2 class settings x 3 seeds x 5 folds.
- Every effective config records `gcnet-random-mask-v1` and
  `nested_gnn_rooted_evidence`.
- Turning the variant back to Flat changes only `osram_meaningful_block`.
- No physical GPU4 use and no interference with the GPU3 ComP queue.
- Final reporting averages folds within seed, then eight rates, then seeds;
  IEMOCAP reports ACC and UA and remains labelled internal Test-oracle.

