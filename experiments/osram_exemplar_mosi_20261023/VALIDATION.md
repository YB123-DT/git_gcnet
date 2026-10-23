# Implementation verification

2026-10-23. This is implementation evidence, not a performance result.

- RED: new module tests first failed because `gcnet_missing_m3.exemplar` did
  not exist. Retrieval-only tests subsequently passed; integration tests moved
  to the accepted remote s0 environment because local comp-repro lacks PyG.
- GREEN: 11 new module/integration tests passed on biggpu CPU in s0
  (Torch2.2.2+cu121). Entire-conversation exclusion, label-use control,
  zero-neighbor safety, gradients, exact disabled output/RNG, missing-current
  and missing-support feature safety, causality, cache invalidation, one scan,
  reload and Dropout RNG protection are covered.
- Seven stdlib runner contract tests passed: finite two-arm seed66 budget,
  original LR schedule, mask/update parity, strict config normalization,
  scope-preserving worker command, artifact-backed completion, and retaining
  the explicit recovery flag until a process is successfully started.
- All14 original `tests/test_osram.py` regression tests passed on biggpu s0.
- Python parse/compile checks and whitespace diff checks passed.
- The only observed test warning is the installed PyG distributed deprecation;
  no dependency or environment was changed.
- Constructed default retrieval module: 272,960 additional parameters.

Real MOSI GPU smoke and formal training are recorded separately in each run's
SUMMARY.json and the queue ledger. They are not claimed by these CPU checks.
Formal jobs must wait until both real-data smokes pass on the frozen source.

Known boundary: feature-only label encoder weights receive zero input and
therefore zero task gradient; its bias remains trainable. This is not a broken
gradient path. The two arms have equal nominal, not equal effective, capacity.
