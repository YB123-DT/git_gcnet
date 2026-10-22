# Six MOSI paired interface experiments

User-approved three A/B hypotheses, seed66, six independent joint-training runs.
Starting commit: c53cae714560f90f6f0dd3598b383c22440bcf9e.

| Pair | A | B | Only varying connection |
|---|---|---|---|
| 1 | topology-full | topology-diagonal | Layer2 source attention connectivity |
| 2 | local-adapter | local-shared | Original vs updated Local into Local Skip |
| 3 | task-pre-norm | task-post-norm | Whether final emotion_norm acts on deltaH |

Reuse original V3 pair encoder, same-role source mixing and four role-query
contexts. Pair1 preserves the original four zero-init read decoders; pair2
concatenates contexts to256 and zero-projects to Local256; pair3 zero-projects
the same256 to1600. Pair2/3 leave original Base/Gap untouched. Task-post-norm
returns LN(original anchor)+deltaH into the one original linear head, hence its
bias is applied once and its weight is shared exactly as requested.

Original CHM, queries, joint writes, causal order, task, data, missing sampler,
Flat capacity and optimizer stay fixed. Each pair has identical initialization
and nominal parameter count. Diagonal source Q/K derivatives are structurally
zero; its effective freedom is not claimed to match dense attention.

MOSI only, seed66,150epochs,3e-4 for100 then3e-5 for50; batch32,Adam,WD1e-5,MSE,
original GCNet random cyclic rates0.0–0.7, saved Flat66 exact evaluation masks,
per-rate Test-best W-F1, all eight BEST plus predictions and full last state.
INTERNAL TEST-ORACLE SCREEN. No other seeds, methods, losses or searches.

Work: standalone core and focused tests; minimal OSRAM/config routing and
formula/gradient integration checks; reuse accepted training/persistence helpers
in independent worker/controller/report; immutable Git source snapshot;
hash-verified default label directory; real GPU smoke; six parallel formal jobs
on healthy biggpu devices with measured-memory admission and persistent tmux.
Do not modify existing runs or checkpoints. Commit and push only task files.
