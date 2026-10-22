# Three-seed supplement launch validation

INTERNAL TEST-ORACLE SCREEN. Source83000b3faa0f56979afb80cdd62351e1a8ef32bd.

Tests-first observed five missing-wrapper/injection failures and two missing
queue/report failures. Leader independently ran58 tests:58 passed in8.73seconds.
These include all new seed restrictions/reference/config tests,existing six-arm
and MOSEI capacity tests,original V3 masks/initialization/gradient regressions,
two-job queue commands and refusal to show partial means as three-seed results.
py_compile and git diff --check passed.

```bash
python -m pytest -q \
 tests/test_mosi_v3_capacity_3seed_worker.py tests/test_mosi_v3_capacity_3seed_queue.py \
 tests/test_mosi_v3_capacity_worker.py tests/test_mosi_v3_capacity_queue.py \
 tests/test_mosei_v3_capacity_runner.py tests/test_source_role_v3_capacity.py \
 tests/test_source_role_v3.py tests/test_source_role_v3_integration.py \
 tests/test_source_role_v3_runner.py
```

Two real MOSI GPU smokes completed,each2epochs/4 genuine optimizer updates:

| Seed | Peak allocated MiB | Seconds | Full model/Adam/RNG/prediction reload |
|---:|---:|---:|---|
| 67 | 3335.96 | 40.70 | exact |
| 68 | 3191.89 | 43.98 | exact |

Each seed validated its own original Flat initialization,config,training seed
protocol,data and full reference history. Common test availability matches the
acceptedseed66 GCNet test bank exactly. V3 zero-init predictions equal same-width
Flat;original/V3 parameters update,gradients are finite/nonzero,and single CHM
scan counts match optimizer steps. Gradient hooks are removed after epoch2.

Both formal jobs are actually running on biggpu physicalGPU2,UUID
GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4,PIDs2981007/2981016. First training
epochs passed original per-seed mask/order/update/LR parity. No extra seed66,
small-Flat-only,other dataset or HPO run was submitted. Existing jobs untouched.

Seed66 source86e3900 completed150epochs,WF179.453246/high73.518892. The new
entrypoint source differs only in default-preserving seed/reference plumbing;
the report explicitly checks core computational file hashes across candidates.
Full three-seed results remain pending;no gain is inferred from smokes.

Detailed commands,environment,reference/mask hashes and live-state snapshot are
in LAUNCH.json. Persistent tmux socket osram_mosi_capacity_3seed/session
mosi_v3_67_68 automatically completes the finite two-run queue and reports:

```text
/data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1
```
