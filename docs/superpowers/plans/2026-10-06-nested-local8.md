# Nested Local eight-node comparison

Goal: one authorized seed66 MOSI100epoch Test-oracle screening, old Nested
64d/3GIN/mean pooling retained except Local becomes8 graph nodes.

Local256 is split contiguously into8x32, each projected32->64 with identity
embeddings. Base/active Gap retain original8x64 actual Memory heads. Local
group indices are new graph coordinates, NOT original OSRAM query heads.
Every Local node connects all active nodes; same-role and same-index edges
remain. Eight Local roots therefore each cover the whole active graph but
different root markers/content. Maximum active nodes32, not25. Graph output
Local nodes each decode64->32 and concatenate256, zero initialized. Original
Local Skip stays unchanged. Only forward512 history reads can change; first
utterance/padding/inactive Gap protections unchanged.

- [x] Native implementation slice: new nested_local8.py, minimal registry
  and build_new40 routing; tests first for split/decode, graph nodes, masks,
  zero bridges and finite updates. Old Nested files untouched.
- [x] Leader: runner choices/config delta only new blockID; canonical ordered
  masks final verification; docs/runtime config with no extra task loss.
- [x] Focused CPU tests including actual task3updates and old regression,
  read-only spec+quality review. No GPU smoke/new dependencies.
- [x] Scoped Lore commit/push; immutable git archive; live healthyGPU6-only
  admission (GPU4 forbidden), durable process launch; verify PID and epoch.
- [x] Save launch evidence. Runner configured to retain all8BEST/predictions/full last_training;
  completion and final scores remain pending.
  No auto multi-seed or hyperparameter sweep. Reuse old Nested and Flat66.
