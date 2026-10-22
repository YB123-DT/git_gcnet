# MOSEI completed capacity grid

INTERNAL TEST-ORACLE SCREEN

Four fresh seed66 runs;two prior1024-width arms reused without retraining.
60epochs,constant1e-3,newGCNetmissing,unchanged task/model computation.
Model/sampler/task core and exact test masks verified across new/reused arms.

| Model | Reused | Scope | ACC | W-F1 | MSE |
|---|---|---|---:|---:|---:|
| f1024_plain | True | mean | 83.235 | 83.078 | 0.622369 |
| f1024_plain | True | high | 80.123 | 79.850 | 0.728433 |
| f1024_v256 | True | mean | 83.286 | 83.104 | 0.630706 |
| f1024_v256 | True | high | 80.095 | 79.820 | 0.744262 |
| original_flat | True | mean | 83.493 | 83.279 | 0.615713 |
| original_flat | True | high | 80.517 | 80.182 | 0.730888 |

## State

- f512_plain: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f512_plain', 'reused': False, 'status': 'running', 'epoch': None, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7032929, 'v3': 0}, 'seconds': None}
- f512_v128: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f512_v128', 'reused': False, 'status': 'running', 'epoch': None, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}, 'seconds': None}
- f512_v256: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f512_v256', 'reused': False, 'status': 'running', 'epoch': None, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 9349567, 'v3': 2316638}, 'seconds': None}
- f1024_v128: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f1024_v128', 'reused': False, 'status': 'running', 'epoch': None, 'error': None, 'parameters': {'flat_adapter': 6106176, 'total': 10912575, 'v3': 831710}, 'seconds': None}
- f1024_plain: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/small_flat', 'reused': True, 'status': 'complete', 'epoch': 60, 'error': None, 'parameters': {'flat_adapter': 6106176, 'total': 10080865, 'v3': 0}, 'seconds': 6149.826286859985}
- f1024_v256: {'path': '/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/wide_v3', 'reused': True, 'status': 'complete', 'epoch': 60, 'error': None, 'parameters': {'flat_adapter': 6106176, 'total': 12397503, 'v3': 2316638}, 'seconds': 8434.529527131002}
- original_flat: {'path': '/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat', 'reused': True, 'status': 'complete', 'epoch': 60, 'error': None, 'parameters': None, 'seconds': 10190.855628709018}

Only completed results shown;missing runs are not zeros. Per-rate Test-best
can select different epochs,not a single-checkpoint independent test score.
One-seed screening only;no automatic HPO or multi-seed expansion.
