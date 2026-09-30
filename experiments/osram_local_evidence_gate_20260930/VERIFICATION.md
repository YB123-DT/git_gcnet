# Verification — 2026-09-30

Parent source: `59d0a10`, branch `feature/osram-uniform-forced-text`.
These checks establish implementation behavior, not sentiment performance.
No formal run, checkpoint selection, or test-label fitting was performed.

## Environment

Server `ssh biggpu` (hostname user22), host GPU0 UUID
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; GPU4 excluded.
Python `/data2/yb/reproduction_workspace/envs/s0/bin/python`.
Snapshot `/data1/yb/remote_experiments/osram_local_evidence_gate_verification_20260930/code`.
Core source hashes were compared between local and snapshot and match:

```
017eab1f8bbe7990627f32997eb45d2ffc5d7a326e58622d779134e2ced4a921  osram.py
91310055471afdb1dd780b0c18e07bcd59bd8ef5126b3e407d759795b1c26156  model.py
e2bc15b2a559b76d0f9080b9a1c8f8c3a0a9f88c9444fdf4f47d3e1bc16a56ed  train_gcnet.py
```

## Direct actual-cfg84 check

Run `python -m tests.verify_local_evidence_gate_cfg84` with the UUID above in
`CUDA_VISIBLE_DEVICES` and `OMP_NUM_THREADS=1`, from the snapshot directory.
The script reads the original seed66 config at
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json`.
It builds fresh paired models (not checkpoint inference), with actual feature
dimensions 512/1024/1024 and output1600.

- Shared initial state exactly equal, original OSRAM parameters trainable.
- Eval and training initial outputs bitwise equal: max_abs 0 in both modes.
- CUDA RNG after each paired forward exactly equal.
- Three Adam updates: finite gradients; Query projector, Flat adapter and Local
  skip weights actually changed.

## Unit/integration checks

New tests first failed on missing gate class/configuration before implementation.
The final new test module passes on biggpu: 11 passed (including five imported
existing Post-GRN tests); one existing PyG deprecation warning.

Checks include explicit off/default equivalence, identity initialization and RNG,
NaN inactive/padding masking, inactive evidence gradient isolation, readout
ablation isolation, active-only penalty reduction, regularizer refresh, separate
type-conditioned coefficients, active-count weighted diagnostics, and actual
two-step `train_epoch` verification of `loss - classification_loss = lambda * R`.
Local syntax compilation and `git diff --check` pass.

Final related regression suite: **63 passed, 1 existing PyG warning, 22.45s**.
Command from the same remote snapshot, UUID binding and environment:

```sh
python -m pytest -q tests/test_local_evidence_gate.py tests/test_history_input_gate.py tests/test_history_input_gate_runner.py tests/test_osram_post_grn.py tests/test_post_grn_integration.py tests/test_memory_shift_residual.py tests/test_memory_shift_integration.py tests/test_osram_history_query.py tests/test_conversation_mixed_masks.py tests/test_completion_memory_write.py
```

Remaining limits: no full MOSI training/evaluation and no evidence of accuracy
improvement. No standalone linter/typechecker configuration exists at repo root;
the verification uses compilation, focused runtime tests and scoped diff review.
