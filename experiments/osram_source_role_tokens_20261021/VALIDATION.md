# Pre-launch implementation verification

INTERNAL TEST-ORACLE SCREEN. Verification is not evidence of task improvement.

## Tests-first record

Remote s0 isolated workspace:
`biggpu:/data2/yb/remote_experiments/source_role_tokens_test_yJKKuU`.
Before implementation, nine new tests failed because the new module/layout
did not exist. After implementation and an additional actual-source causal test:

```
64 passed, 1 deselected, 1 warning in 8.12s
```

Command (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1):

```bash
OSRAM_LEGACY_SOURCE=legacy/gcnet_missing_m3/osram.py \
/data2/yb/reproduction_workspace/envs/s0/bin/python -m pytest -q \
 tests/test_source_role_tokens.py tests/test_source_role_tokens_runner.py \
 tests/test_source_relation.py tests/test_source_relation_integration.py \
 tests/test_source_relation_runner.py tests/test_source_differentiable_base.py \
 tests/test_source_role.py tests/test_source_role_direct.py \
 -k 'not test_residual_default_keeps_previous_module_exact' --disable-warnings --tb=short
```

The one historical source-role regression needs `git show f6bc52e` and cannot
run in the Git-free immutable test copy. It was explicitly deselected, not
claimed passed. The legacy Flat regression DID run by loading the exact
`8cde443` original `osram.py` exported from Git into the stated legacy path.
An initial additional CLI assertion omitted pre-existing required feature/output
arguments; its fixture was corrected and the complete command rerun green.
The warning is from installed third-party packages; warnings were not counted
as failed tests.

Local pure runner tests:12 passed; Python compilation and `git diff --check`
passed. AST comparison against297e588 confirmed unchanged `block_write`, `_scan`,
`_address_residual`, `_project_sequence` bodies.

## Mechanism checks

- Attention receives12 role-major tokens; opposite changes to two Gap slots
  cannot vanish before Attention through averaging.
- Separate slot identities/masks; no-history/all-masked rows skipped;
  inactive NaNs cannot leak into outputs or gradients.
- Every12 source-read input slot receives finite nonzero gradients in the
  activated synthetic test; genuine tracked reads also receive gradients.
- Original initialization, training RNG, Flat hidden/contexts and padding
  match at zero initialization; old source-mean tests remain green.
- Actual TrainConfig/model factory and actual CLI parser route role-tokens.
- Checkpoint serialization/reload exact; modifying future input cannot alter
  prior outputs through the new branch.

## Parameter counts and environment

Measured in s0 (Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2):
new module530,470 parameters; old source-mean553,678. Full model count is
expected14,040,263 from unchanged original13,509,793 and is checked again
in real-data GPU smoke.

Independent read-only review found no implementation blocker. Its important
scope warning is retained in README: this changes token encoding and parameter
count, not merely one mean operator. It cannot prove why the old model declined.

GPU smoke and training evidence are pending at this source commit. Record
actual immutable SHA, smoke output, process identifiers and results separately
after execution. Do not fill scores from expectations.
