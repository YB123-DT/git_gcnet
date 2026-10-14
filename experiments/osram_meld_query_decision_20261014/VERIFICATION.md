# Implementation verification

INTERNAL DIAGNOSTIC ONLY. Correct implementation is not evidence of a gain.

Tests-first: core learning tests failed12/12 for missing implementation; trace
tests failed5/5; runner tests failed4/4 for missing runner; checkpoint identity
test failed for missing binder. Final remote CPU suite: **24 passed** (4.06s),
plus trace implementer's8existing related capture/replay tests passed. Local
`py_compile` and `git diff --check` passed. No CHM/model/training backbone edits.

Bounded real-MELD GPU smoke: host GPU0, exact source seed66 rate0.7, one training
batch298utterances / one test batch293utterances, twoepochs per diagnostic stage.
All five fits completed in4.09s after loading/hash checks; peak allocated CUDA
438,385,664bytes. Source state hash unchanged, exact saved classes/labels on the
partial test replay, all donor rows matched. Not a full-test score or performance
claim. Readout152,781trainable params; corrector54,016trainable params.

C_real/C_donor initial maximum B logit difference4.768e-7 with identical classes;
both frozen decoder before/after hashes match. Both query fits4/4steps had finite
nonzero owner gradients. On nonempty-history eligible heads, all recorded Query
gradients finite/nonzero: Base4256, Gap-A2848, Gap-T2768, Gap-V2896 each. Inactive
queries and empty-memory behavior are separately locked by unit tests.

First GPU smoke failed at best-weight reload because Torch1.13's NVML UUID
enumeration reported0devices while its CUDA runtime reported1 and computed on
cuda:0. Minimal reproduction confirmed this independent of our model. Fix:
load saved storages on CPU then copy `load_state_dict`; seed/save RNG explicitly
for admitted logicalcuda:0, rather than enumerate devices. GPU UUID admission
and hostGPU4 prohibition remain intact. Second smoke succeeded; first failure
directory retained, not overwritten.

Read-only specification and integration reviews checked separate Query optimizer,
immutable selected B decoder, donor state with recipient Q/Key, label-free donor
matching, and common populations. Review fixes bind weights/config/epoch to the
selection manifest, masks/data to immutable artifact SHA inventory, save frozen
decoder hashes and per-slot Query-gradient coverage, and reject mixed budgets
or source protocols in aggregation. No automatic multi-seed/D control/search.

CPU command (biggpu staged directory):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. \
  /data2/yb/conda_envs/cmerc-repro/bin/python -m pytest -q --tb=short \
  tests/test_meld_query_decision_learning.py \
  tests/test_meld_query_decision_trace.py \
  tests/test_meld_query_decision_runner.py
```

Successful GPU artifact:
`/data2/yb/remote_experiments/osram_meld_query_decision_20261014_smoke_v2/SUMMARY.json`.
At implementation commit16d0313 the full result was pending. All eight rates
subsequently completed (40fits); full data/checks are in runs/*.json and
INTERPRETATION.md. No source update or frozen-decoder violation occurred.
