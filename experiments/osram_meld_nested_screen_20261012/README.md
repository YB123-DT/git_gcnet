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

After the 70-run screen completes, `followup_top7.py` ranks the 64 seed-66 grid
configurations by 8-rate mean ACC and launches seeds 67 and 68 for the top
seven. This produces exactly 14 concurrent follow-up runs, seven on each GPU,
and completes a three-seed result together with the existing seed-66 run.

`dispatch_local14.py` is a separate one-shot seed-66 refinement on physical
GPUs 2 and 3. It fills ten missing cells in the compact 32/64-d local cube and
adds four 48-d midpoint configurations. None duplicate the original 64-grid
screen; checkpoints are discarded after metrics are durable.
