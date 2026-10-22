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

## Remote deployment verification

Immutable runtime code commit: `d00d0629d3cca484526cc9718a6ade883925ca45`,
deployed to `/data2/yb/remote_experiments/osram_v3_connection_screen_20261022/source_d00d062`.
Core module, OSRAM and worker SHA256 matched the committed local bytes. The actual
s0 environment passed **65 tests in 8.96 s** (new core, integration and runner;
the Git-history test was excluded because the deployed archive has no `.git`).

`execution_v1` stopped during the first smoke before any optimization because the
code-only archive omitted the default `dataset/` label path. The original import
chain resolves this path before the later environment override. This was a
deployment omission; the error log and failed queue remain intact. Provided a
link to the existing verified dataset directory, checked both pickle hashes
against their original manifest paths, and launched independent `execution_v2`
from the same code commit. No model, mask or training hyperparameter changed.

The controller is persistent under tmux socket `osram_v3_connections`, session
`screen_v2`. It uses physical GPU2 / GPU7 by their verified UUIDs and admits at
most three formal jobs per card according to measured smoke peaks. Actual final
smoke states and formal PIDs are captured in `LAUNCH.json` after observation.

## Six real-data CUDA smokes

All six completed successfully in the original s0 environment. Each performed
four actual Adam updates, verified finite nonzero task gradients in all required
variant parameter groups and the source-read inputs, checked original and added
parameter updates, exact original initialization/RNG, eight-rate first-batch
initial prediction equality, and optimizer/RNG/prediction equality after full
checkpoint reload. Original reference masks/configuration/data were accepted.

| Dataset | Variant | Smoke peak allocated MiB | Total parameters |
|---|---|---:|---:|
| MOSI | head-preserving | 4544 | 13,845,311 |
| MOSEI | head-preserving | 3244 | 13,845,311 |
| MOSI | source-slots | 3300 | 14,024,511 |
| MOSEI | source-slots | 2046 | 14,024,511 |
| MOSI | flat-hidden | 3366 | 14,121,791 |
| MOSEI | flat-hidden | 2334 | 14,121,791 |

The six formal jobs were then admitted by the controller. Smoke scores are not
reported as model performance; training results remain running until their full
150/60-epoch budgets and selected-checkpoint integrity checks finish.

At `2026-10-22T17:06:37Z`, all six formal processes were alive with command lines
matching their queue records, had passed the four-update gradient check, and had
saved formal epoch records. MOSI head/source/hidden were at epochs19/16/14;
MOSEI all at epoch1. Every saved last training row reported exact reference mask
trace/update/LR parity and one CHM call per optimizer update. Snapshot PID/GPU
assignments and current formal memory peaks are in `LAUNCH.json`; formal peaks
can exceed the limited smoke measurements, and are not presented as isolated
benchmark measurements. No existing jobs were signalled or changed.

## Completed result check (2026-10-22 22:43 UTC)

The controller and all six workers report `complete`, with MOSI150/MOSEI60 fully
finished and no errors. Each run retains eight BEST checkpoints and corresponding
predictions. All 48 BEST checkpoint paths were present, and all 48 selected
prediction SHA256 values matched their ledgers. The generated report contains
64 per-rate rows (six candidates plus two existing Flat references), and sixteen
mean/high aggregate rows. Completed reports are synced as RESULT.md, REPORT.json
and PER_RATE.csv; LAUNCH.json remains the original observed launch snapshot.

Head-preserving is the best candidate on both datasets, but eight-rate W-F1 gains
are only +0.082274 pp on MOSI and +0.013645 pp on MOSEI. Its high-missing gains are
+0.028592/+0.006054 pp respectively. Source-slots and flat-hidden decline on both
datasets. This single-seed internal Test-oracle screen supports near-parity for
the head-preserving connection, not a demonstrated stable performance gain.
No additional seeds, HPO or new experiments were automatically launched.
