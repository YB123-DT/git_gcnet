# MOSI capacity screen implementation / launch validation

INTERNAL TEST-ORACLE SCREEN. No formal performance result is established yet.

Immutable source: `86e3900d7fb5f3ee03ce214362e0fececb42887e`.

## CPU

Tests-first: missing worker produced four expected failures; missing queue/report
produced three expected failures. After implementation the leader ran all seven
relevant files:51 passed in9.09seconds. The subagent separately verified29 tests.

```bash
python -m pytest -q \
 tests/test_mosi_v3_capacity_worker.py tests/test_mosi_v3_capacity_queue.py \
 tests/test_mosei_v3_capacity_runner.py tests/test_source_role_v3_capacity.py \
 tests/test_source_role_v3.py tests/test_source_role_v3_integration.py \
 tests/test_source_role_v3_runner.py
```

Coverage includes only-six-arm config changes,invalid protocol rejection,LR
boundary at100/101,smoke length,cumulative gradient evidence and hook removal,
resident-peer memory reservation,completed-only reporting,and unchanged core V3
defaults/masks/initialization/RNG/output/gradients. py_compile/diff check passed.

## Real MOSI GPU checks

All six independent real-data smokes completed successfully. Each executed two
epochs,four original batch32 optimizer steps,not synthetic feature fitting.
Original and V3 task parameters changed; monitored V3 groups became finite and
nonzero; exactly one CHM scan per training batch. Hooks were removed after their
first-two-epoch validation. All original task and mask computations stayed intact.

| Mode | Peak allocated MiB | Optimizer steps | Outcome |
|---|---:|---:|---|
| f512_plain | 1147.80 | 4 | complete |
| f512_v128 | 3190.67 | 4 | complete |
| f512_v256 | 3371.67 | 4 | complete |
| f1024_plain | 1189.97 | 4 | complete |
| f1024_v128 | 3237.30 | 4 | complete |
| f1024_v256 | 3414.96 | 4 | complete |

Every arm verified exact zero-init predictions against its same-width Flat,
unrelated original initialization,original reference config and dataset/split,
exact original test-mask hashes and train-mask/order/step/LR parity. Complete
model/Adam/RNG restoration reproduced predictions/metrics exactly. Checks use
archived s0 Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2.

Small-data smoke metrics are not training results. All smoke checkpoints remain
in their independent directories. Formal workers start from fresh initialization,
not these four-update smoke weights.

## Launch

Persistent controller PID2785369,tmux socket `osram_mosi_capacity`,session
`mosi_capacity`; physical GPU3 and7,three formal processes per GPU,measured-peak
admission. Six formal worker PIDs:2798169,2798180,2798184,2798214,2798217,2798221.

```text
/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1
```

Actual commands,device UUIDs,environment,parameter counts and mask hashes are in
LAUNCH.json and the remote QUEUE.json. Existing MOSEI workers and all historical
results/checkpoints were left untouched; GPU4 is excluded.
