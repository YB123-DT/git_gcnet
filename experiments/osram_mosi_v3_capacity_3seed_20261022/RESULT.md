# MOSI Flat512 + V3-128 three-seed verification

INTERNAL TEST-ORACLE SCREEN

Reuse seed66; only67/68 newly trained.150epochs,3e-4→3e-5,new GCNet missing.
Only completed candidates enter metrics; no two-seed mean presented as three seeds.

| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|
| 66 | original_flat | 79.268 | 79.249 | 73.374 | 73.410 |
| 66 | flat512_v3_128 | 79.497 | 79.453 | 73.476 | 73.519 |
| 67 | original_flat | 79.478 | 79.402 | 73.984 | 73.777 |
| 67 | flat512_v3_128 | 78.982 | 78.939 | 72.663 | 72.483 |
| 68 | original_flat | 79.268 | 79.194 | 73.222 | 73.228 |
| 68 | flat512_v3_128 | 79.668 | 79.551 | 73.526 | 73.395 |

Three-seed original WF1: 79.282 ± 0.108; high 73.471 ± 0.279 (sample SD).

Three-seed candidate WF1: 79.315 ± 0.329; high 73.132 ± 0.566 (sample SD).

## States

- seed66: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1/runs/f512_v128', 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}
- seed67: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1/runs/seed_67', 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}
- seed68: {'path': '/data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1/runs/seed_68', 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'flat_adapter': 3058240, 'total': 7864639, 'v3': 831710}}

This candidate was selected using seed66 test scores. Test-oracle and candidate
selection bias remain. No independent-generalization or structural-causality claim.
The small-Flat-only67/68 controls are not trained in this supplement.

## Completion interpretation

All three candidates completed150epochs. Eight-rate WF1 is79.314544±0.328542
versus original largeFlat79.281728±0.108085:only+0.032816pp. High-missing WF1
is73.132100±0.565944 versus73.471485±0.279397:-0.339385pp. These are sample SDs,
not confidence intervals or statistical-significance evidence.

The paired eight-rate deltas are+0.204160(seed66),-0.462975(seed67),
+0.357263(seed68). High-missing deltas are+0.109100,-1.294034,+0.166780pp.
Thus the seed66 improvement does not establish a stable three-seed performance
gain. The candidate reduces parameters13.51M→7.86M(about42%),which is a separate
capacity observation,not proof of improved historical evidence utilization.

COMPLETED.json records successful controller exits,complete150epoch summaries,
selected epochs,summary/history hashes and presence of all18 expected checkpoint/
mask/prediction artifacts per candidate. All24 selected prediction-file hashes
were verified. No new inference,training,weight deletion or automatic expansion.
