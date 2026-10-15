# Checkout isolation

This delivery directory is copied from the local experiment implementation. Its default run root is `<this checkout>/runs/baseline_5models_local_20261015`, including dataset caches, teachers, checkpoints and logs. It does not reuse or modify `/data2/yb/paper/runs/baseline_5models_local_20261015`. Set `BASELINE_RUN_ROOT` only to another private output directory.

`PAPER_WORKSPACE` selects read-only input assets; `BASELINE_PYTHON` selects the existing environment. The host restriction (`user23`), two GPU UUID whitelist, no-upload environment, one-job-per-GPU limit and file locks remain enforced. Never point this delivery copy at the running experiment output root or start it alongside the active local dispatcher on the same GPUs.

No model parameters, loss functions, missing-mask sampling rules or checkpoint-selection rules changed during packaging. Delivery fixes to restoration/provenance are documented in REVIEW_GUIDE.md. Running experiments retain their original immutable snapshots.
