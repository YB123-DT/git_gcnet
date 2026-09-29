# Pre-launch verification

biggpu host GPU0, original s0 environment, 2026-09-29.

- New core tests first failed on missing HistoryInputGate; four new tests
  then passed. Launcher tests first failed on missing runner, then passed.
- Related remote regression suite: 52 passed, one pre-existing PyG
  deprecation warning. Includes gate core/runner, post-GRN, memory-shift,
  history-query, conversation masks and completion-memory paths.
- Independent real cfg84 CUDA check: load original seed66 config; build
  Flat and gate variant with actual dimensions512/1024/1024 and output1600.
  Shared parameters match exactly. Eval AND train initial outputs match
  bitwise (max_abs=0), post-forward CUDA RNG matches, alpha mean=1.
  Three joint Adam updates have finite gradients.
- One-epoch MOSI smoke seed66 complete, eight best checkpoints and all
  eight test rates verified against canonical baseline mask protocol.
  JEPA loss0. Train alpha mean1; miss.7 eval alpha mean .99944824,
  boundary fraction0. This is implementation evidence, not performance.
- Diagnostic alpha moments use float64 so near-one variance is not lost
  to float32 subtraction; this detached arithmetic does not change model
  outputs or gradients. Smoke provenance records the preceding source hash.

No original model parameters frozen. No test-label oracle tables loaded.
Core implementation and Flat default-path diff reviewed by the leader.
Source changes exclude unrelated gcnet/model.py edits. Weight files remain
on biggpu; only compact smoke configuration/metrics/provenance are archived.
