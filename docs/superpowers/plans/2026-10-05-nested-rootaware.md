# Nested root-aware pooling implementation plan

Required execution skill: executing-plans; focused TDD before production edit.

Goal: user's approved three-seed variant of existing Nested GNN, changing only
each layer's subgraph pooling. Original current graph/topology/three GIN-style
layers/root marking/Flat adapter-only residual/zero-init bridges/task loss remain.
Local Skip remains original (the existing adapter does not alter Skip).

New pooling: concat(root hidden, non-root neighbor mean) -> shared Linear128,64.
An empty non-root set gives exactly zero neighbor vector. Same projection reused
at all three depths. No extra activation, attention, gate or auxiliary loss.
New method ID nested_gnn_rootaware_evidence; preserve original 40-item catalog
via a separate variant family. Initialization of added projection is RNG-forked
so common weights and downstream initialization keep original sequence.

- [ ] Tests first: observed old absence of variant, root selection/neighbor
  exclusion, singleton zeros, projector shared/count8256, old parameter/RNG
  preservation, safe padding/inactive NaNs, initial identity and finite updates.
- [ ] Implement root-aware option and separate registry variant; old method
  remains exactly the original mean pooling path.
- [ ] Main integrates variant family and runner's fixed NEW METHOD selection;
  add dedicated three-seed dispatcher/report, reuse hash/data/resume runner.
- [ ] One focused remote CPU test run, actual parameter count, read-only review.
- [ ] Commit/push, seal code, GPU6 admission; launch three independent runs with
  exact same-seed Flat protocol and verify true optimizer updates/history.
- [ ] Record launch commands/PIDs/UUID and push; automatic complete summary
  includes old Nested and Flat at matched seeds, all per-rate scores and failures.

Ownership: isolated block agent owns meaningful_new40_structure.py,
meaningful_new40_registry.py, meaningful_input_new40.py and dedicated tests.
Main owns meaningful_input.py, runner/dispatcher/report; no unrelated edits.

Validation command: existing biggpu s0 Python -m pytest -q
tests/test_nested_rootaware.py plus old catalog regression. GPU6 only,
GPU4 forbidden. Three seeds66/67/68,100epochs each; eight BEST Test-oracle
checkpoints and predictions/full last_training.pt. INTERNAL DIAGNOSTIC ONLY.
