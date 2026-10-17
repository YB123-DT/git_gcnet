# Isolated code delivery plan

1. Keep all code under experiments/baseline_5models_local_20261015 in a new branch from origin/main.
2. Preserve baseline mathematics and frozen model configuration. Change only delivery paths/test fixtures to keep a fresh checkout isolated from active runs.
3. Vendor two traceable MoMKE reference files for offline architecture-equivalence tests.
4. Exclude live registries, masks, predictions, metrics, logs, datasets, checkpoints and caches.
5. Run CPU regression, syntax and fresh-registry checks in the delivery worktree; do not start GPU jobs from the delivery copy.
6. Check exact destination URL and outgoing path/size list; push only the isolated branch to YB123-DT/git_gcnet.

## 2026-10-17 explicit result-publication override

The user explicitly authorized uploading missing results. Publish only the reviewed dynamic result snapshot and clearly classified historical scores under published_results. Keep weights, features, prediction/mask arrays and full training logs local. The interrupted withdrawn draft remains untracked and must not be staged. No unrelated repositories or branches are touched.
