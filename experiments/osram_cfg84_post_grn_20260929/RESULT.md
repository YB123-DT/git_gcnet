# cfg84 Flat + post-GRN: completed three-seed comparison

INTERNAL TEST-ORACLE DIAGNOSTIC; NOT A FORMAL PAPER RESULT.

Seeds 66/67/68 each completed 100 epochs on biggpu host GPU 0. Original
Flat references were reused, not retrained. Random-missing cyclic training,
no-JEPA, Flat readout, output_dim=1600, one-stage joint optimization.
The experimental switch is `osram_post_grn=True`; no persistent mixture.

| W-F1 (%) | Flat | Flat + post-GRN | Paired delta (pp) |
|---|---:|---:|---:|
| Eight-rate mean | 80.559 ± 0.507 | 79.521 ± 0.177 | −1.039 ± 0.381 |
| High missing (.5/.6/.7) | 75.594 ± 1.016 | 74.665 ± 0.165 | −0.929 ± 0.897 |

Uncertainty is sample standard deviation across three seeds. All three
eight-rate seed means decrease. This is not evidence of improvement.

Source of numbers: `results/SUMMARY.json`. Per-seed configuration, history,
metrics, diagnostics and provenance are retained in `results/seed_*/`.
Checkpoints and predictions remain on biggpu at
`/data1/yb/remote_experiments/osram_cfg84_post_grn_20260929/runs/`.
The isolated training snapshot is the sibling `code/` directory. Source
hashes in provenance, rather than the subsequent archival commit, identify
the code actually run. `source-dirty.patch` captures accumulated core edits.

Selected-checkpoint gate means range approximately .435–.474, with nearly
zero saturation. Residual/Flat norm ratios range approximately .072–.482.
The branch is active; these magnitudes alone do not identify why accuracy
decreased or prove that its residual causes the decrease.

The archival source commit also preserves earlier opt-in completion-write,
history-query and memory-shift code already present in the worktree. These
options are disabled for this post-GRN comparison. Unrelated GCNet changes,
feature caches and checkpoint binaries are not part of the archive.

## Archival verification caveat

At archival time, local `unittest` execution of the three post-GRN test
modules ran 17 tests: 16 passed, but the CUDA full-model exact-equality
assertion failed. Earlier remote verification passed. This environment
discrepancy remains open at this archival checkpoint; do not claim all
current verification passed or attribute the score drop to architecture
alone before investigating the discrepancy. No training is restarted.
