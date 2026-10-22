# MOSEI grid implementation and launch evidence

INTERNAL TEST-ORACLE SCREEN. Formal scores for the four new arms are pending.

Immutable training source:7ec78786b12b6006bd8f6ebcfab32ee9f6026a23.

Tests-first:five expected missing worker/injection failures,three expected missing
queue/report/injection failures. Leader independently ran66 focused/regression
tests:66 passed in8.44seconds. py_compile and git diff --check passed.

```bash
python -m pytest -q \
 tests/test_mosei_v3_grid_worker.py tests/test_mosei_v3_grid_queue.py \
 tests/test_mosi_v3_capacity_3seed_worker.py tests/test_mosi_v3_capacity_3seed_queue.py \
 tests/test_mosi_v3_capacity_worker.py tests/test_mosi_v3_capacity_queue.py \
 tests/test_mosei_v3_capacity_runner.py tests/test_source_role_v3_capacity.py \
 tests/test_source_role_v3.py tests/test_source_role_v3_integration.py \
 tests/test_source_role_v3_runner.py
```

Tests cover four-new-arm restrictions,no duplicate old configurations,correct
parameter budgets and protocol,rejectotherseeds,original two-arm default behavior,
finite/removable gradient hooks and preserved scan hooks,shared queue defaults,
incomplete results not scored,and existing V3/othercapacity regressions.

## Actual real MOSEI GPU smokes

All four independent smokes completed,eachthree genuine batch32 Adam steps.
Verified zero-init equality to same-widthFlat,untouched original initialization,
original feature/split/config/mask hashes,finite/nonzero V3 gradient groups,
single CHM scan,model/Adam/RNG/prediction restoration. Smoke weights remain
separate and are NOT initialization for formal training.

| Mode | Actual total params | Peak allocated MiB |
|---|---:|---:|
| f512_plain | 7,032,929 | 566.90 |
| f512_v128 | 7,864,639 | 2173.20 |
| f512_v256 | 9,349,567 | 2180.05 |
| f1024_v128 | 10,912,575 | 2212.20 |

The full model/sampler/task core hashes,test masks,split,seed protocol and data
files were directly compared across four new smoke identities and both reused
completed1024-width runs:all match. Entrypoint/instrumentation hashes differ
and are recorded,not concealed. Old two-arm defaults are preserved;only new
runs remove V3 diagnostic hooks after first-epoch connectivity validation.
This is a logging overhead difference,not a performance claim or optimizer change.

## Actual launch

Four formal processes actually running on two healthy biggpu GPUs:

| Mode | Physical GPU | PID |
|---|---:|---:|
| f512_plain | 1 | 3235085 |
| f512_v128 | 2 | 3235093 |
| f512_v256 | 1 | 3235088 |
| f1024_v128 | 2 | 3235104 |

Both devices are bound by real UUID,worker cuda:0 is logical only. GPU4 is
excluded. No other process was signalled. Expected config60epochs,constant1e-3,
batch32,Adam,WD1e-5,newGCNetmissing and original task all verified. Every full
epoch will check original mask/order/update/LR parity;no completed new score
or speed improvement is claimed at this launch snapshot.

Persistent tmux socket osram_mosei_grid/session mosei_grid automatically finishes
the finite four-job queue and reports complete-only metrics,including reused rows:

```text
/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1
```

LAUNCH.json stores exact commands,PIDs,device UUIDs,environment,mask/reference
hashes and smoke restoration fields. All existing/new checkpoints,smokes,masks,
predictions and logs are preserved. No HPO/extra seeds or old run retraining.
