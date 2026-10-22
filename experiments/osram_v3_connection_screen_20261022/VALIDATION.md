# Verification ledger

INTERNAL TEST-ORACLE SCREEN. Implementation correctness is not performance evidence.

## Local implementation validation (2026-10-22)

Starting source: `614799a24e910644f8dc5cf84bd58a034b0df212`.

Tests-first: the new block tests failed because the dedicated implementation did
not exist; integration tests then failed eight expected constructor checks for
the missing `osram_source_role_v3_variant` argument. An initial collection attempt
in `msa_extract` lacked `torch_geometric`; no dependency was installed. Integration
uses the existing `multimodalerc310` Python/Torch environment, appending only the
existing `msa_extract` site directory to resolve pytest after native packages.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c '
import sys
sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages")
import pytest
raise SystemExit(pytest.main(["-q",
"tests/test_source_role_v3_connections.py",
"tests/test_source_role_v3_connections_integration.py",
"tests/test_source_role_v3.py",
"tests/test_source_role_v3_integration.py",
"tests/test_gcnet_legacy_random_mask.py", "--tb=short"]))'
```

Observed: **80 passed in 8.45 s**, Torch `2.2.2+cu121`, CPU only. This includes:

- Old default-off and standard-V3 state/output/gradient/RNG equality, strict legacy loading.
- New branch initialization does not consume downstream model RNG.
- Fresh complete-model predictions exactly equal Flat for each variant.
- Four real optimizer updates on synthetic inputs, finite nonzero new gradients,
  parameter changes, strict checkpoint reload, no temporal/batch leakage.
- Fixed source slots, independent original Memory heads, same-role source attention,
  inactive/padding/ATV/no-history hard masks and NaN-safe inactive inputs.
- Hidden injection before original GELU, unchanged original Flat input and Local Skip.
- Original GCNet missing sampler regression tests.

Measured default module parameter counts:

| User option | Variant | Additional parameters |
|---|---|---:|
| 1 | head-preserving | 335,518 |
| 2 | source-slots | 514,718 |
| 3 | flat-hidden | 611,998 |

The same command with `tests/test_v3_connection_screen.py` also included passed
**90 tests in 7.96 s**. Its additional ten checks cover exactly six authorized
jobs, unchanged training configuration, explicit schema migration, two learning
rate schedules, four-update smoke budgets, failed-GPU rejection, measured memory
admission, actual variant-specific gradient groups, and exclusion of incomplete
or smoke runs from result tables. `git diff --check` and Python compilation passed.

Flat-hidden uses ordinary nonzero `P` initialization and the original zero-initialized
Flat final layer. Initial prediction equality applies to **from-scratch joint
training**, not to injection into a trained nonzero Flat. First-step gradients are
intentionally not identical. Runtime partial classifier ablations bypass the
entire coupled hidden branch to prevent removed evidence leaking back in.

## Remote preflight

Server `biggpu`, existing `/data2/yb/reproduction_workspace/envs/s0/bin/python`:
Python `3.10.20`, Torch `2.2.2+cu121`, NumPy `1.24.3`, pytest `8.3.5`; CUDA available.
No environment or package changes. The accepted Flat reference summaries are
complete at MOSI150/MOSEI60 with their saved masks present. Dataset identities,
configuration, initialization, exact evaluation mask hashes and per-epoch training
mask/update/LR parity are rechecked by each worker before/during training.

GPU smoke, scheduler verification and actual training launch are recorded in
subsequent ledger entries and `LAUNCH.json`; they have **not** been inferred from
the CPU tests above.

## Independent review and final local gate

Separate read-only reviews checked architecture and experiment control. Two
issues were corrected: incompatible nonstandard V3 plus disabled Local Skip now
raises instead of silently ignoring the configuration; restoring a final-epoch
`last.pt` synchronizes the summary epoch/BEST ledger even if the training loop is
empty. The queue saves its final state before publishing the report.

Final combined suite after these changes: **91 passed in 7.74 s**. The Local Skip
rejection and stale-summary recovery checks were observed failing before their
fixes. Remote real-data GPU verification remains required before formal launch.
