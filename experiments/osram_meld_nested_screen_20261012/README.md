# MELD 50-epoch Flat/Nested screen

`INTERNAL DIAGNOSTIC ONLY`: every run uses per-rate Test-oracle ACC selection.

- Six anchors: Flat/Nested, seeds 66/67/68.
- Sixty-four seed66 Nested configurations from a balanced four-level grid over
  node dimension, GIN depth, memory grouping, and Nested MLP width.
- Five waves; every wave launches seven processes on physical biggpu GPU0 and
  seven on physical GPU1. GPU4 is excluded.
- All runs use cfg84 no-JEPA, cyclic random missing, 50 epochs, and the official
  ECERC MELD train/dev/test split.
- Per-rate checkpoints are deleted after metrics/history/provenance are written;
  the selected configuration must be rerun before downstream diagnostics.

The grid is a screening design. Its test-selected scores must not be presented
as an unbiased estimate of the selected hyperparameter configuration.
