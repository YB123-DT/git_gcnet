# Pre-launch verification

INTERNAL TEST-ORACLE SCREEN

Local verification, 2026-10-20, cmerc-xp-repro environment:

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q \
 tests/test_source_role_direct.py tests/test_source_role.py \
 tests/test_source_role_integration.py tests/test_source_role_runner.py \
 tests/test_tasr_integration.py tests/test_tasr_primitives.py \
 tests/test_mosi_gcnet_testbest_worker.py tests/test_mosei_scalar_helpers.py
```

Result: 64 passed. An optional torch-sparse import warning reports missing
libcusparse.so.11; these tested paths do not use that optional operator.
Compileall and git diff --check passed. The direct-mode regression tests were
written before the implementation and first failed for the absent interface.
Two subsequent fixture-only corrections represented the real causal config
and the JSON-persisted historical config; production guards were not weakened.

Verified: unit source weights at zero scores; exact 2*sigmoid formula;
no source Softmax; no original Base/Gap add-back; shared parameter count;
finite new-parameter gradients; poisoned inactive inputs/padding safely masked;
raw Local/contexts unchanged; initialization prediction tolerance; reload;
same initialization RNG; historical residual implementation exact regression;
explicit mode/config guard; one MOSI seed66 job only. Existing source-tracker,
TASR integration, selected-checkpoint and scalar helper tests also passed.

Independent read-only review found no remaining high/medium implementation
blockers after strengthening completed residual comparison config, per-epoch
mask/update/LR checks and reference hashes. Review did not run remote jobs.

Real MOSI GPU smoke is a mandatory separate launch gate. It will record actual
source reconstruction/prediction errors, optimizer updates, checkpoint reload,
memory and time. CPU tests are not evidence of performance improvement.
The formal run compares all 150 training mask/update/LR traces with the
completed Flat reference. Direct-versus-residual final reporting additionally
requires complete, matched 150-epoch histories and matching initialization,
split, seed and evaluation masks; only the source-role mode may differ.
