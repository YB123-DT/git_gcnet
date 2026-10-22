# MOSI Flat512 + V3-128 three-seed verification

INTERNAL TEST-ORACLE SCREEN

Reuse seed66; only67/68 newly trained.150epochs,3e-4→3e-5,new GCNet missing.
Only completed candidates enter metrics; no two-seed mean presented as three seeds.

| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|
| 66 | original_flat | 79.268 | 79.249 | 73.374 | 73.410 |
| 66 | flat512_v3_128 | 79.497 | 79.453 | 73.476 | 73.519 |
| 67 | original_flat | 79.478 | 79.402 | 73.984 | 73.777 |
| 68 | original_flat | 79.268 | 79.194 | 73.222 | 73.228 |

Three-seed original WF1: 79.282 ± 0.108; high 73.471 ± 0.279 (sample SD).

## States

- seed66: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1/runs/f512_v128', 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}
- seed67: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1/runs/seed_67', 'status': 'running', 'epoch': 2, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}
- seed68: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1/runs/seed_68', 'status': 'running', 'epoch': 1, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}

This candidate was selected using seed66 test scores. Test-oracle and candidate
selection bias remain. No independent-generalization or structural-causality claim.
The small-Flat-only67/68 controls are not trained in this supplement.
