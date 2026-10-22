# MOSI head48 three-seed supplement

INTERNAL TEST-ORACLE SCREEN

Reuse completed seed66; train only seeds67/68. 150epochs; LR3e-4 (1–100),3e-5 (101–150).
GCNet random-mask-v1; cyclic0.0–0.7; per-rate Test-best W-F1, earliest tie.
Eight per-rate checkpoints may differ: not single-checkpoint deployment performance.
Head count8, write_step0.6, Flat1600; K/V dimensions48 instead of64.
11,245,857 parameters. Forward context and Flat input width also shrink;
this is not a parameter-matched storage-only ablation.
Only completed150-epoch candidates enter scores. No two-seed mean is called three seeds.

| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|
| 66 | original_flat | 79.268 | 79.249 | 73.374 | 73.410 |
| 66 | head_48 | 79.554 | 79.497 | 74.035 | 73.894 |
| 67 | original_flat | 79.478 | 79.402 | 73.984 | 73.777 |
| 67 | head_48 | 79.192 | 79.085 | 73.018 | 72.838 |
| 68 | original_flat | 79.268 | 79.194 | 73.222 | 73.228 |
| 68 | head_48 | 79.287 | 79.297 | 73.730 | 73.799 |

## Same-seed differences

| Seed | ΔACC | ΔW-F1 | High ΔACC | High ΔW-F1 |
|---:|---:|---:|---:|---:|
| 66 | +0.286 | +0.248 | +0.661 | +0.484 |
| 67 | -0.286 | -0.317 | -0.965 | -0.938 |
| 68 | +0.019 | +0.103 | +0.508 | +0.571 |

## Three-seed original (mean ± sample SD)

mean: ACC 79.338 ± 0.121; WF1 79.282 ± 0.108
high: ACC 73.526 ± 0.403; WF1 73.471 ± 0.279

## Three-seed candidate (mean ± sample SD)

mean: ACC 79.345 ± 0.188; WF1 79.293 ± 0.206
high: ACC 73.594 ± 0.522; WF1 73.510 ± 0.584

## Per-rate Test-best

| Seed | Model | Rate | Epoch | ACC | W-F1 | MSE |
|---:|---|---:|---:|---:|---:|---:|
| 66 | original_flat | 0.0 | 48 | 87.348 | 87.251 | 1.018859 |
| 66 | original_flat | 0.1 | 52 | 83.994 | 83.871 | 1.204979 |
| 66 | original_flat | 0.2 | 49 | 82.622 | 82.595 | 1.342931 |
| 66 | original_flat | 0.3 | 50 | 80.793 | 80.741 | 1.306329 |
| 66 | original_flat | 0.4 | 62 | 79.268 | 79.305 | 1.534990 |
| 66 | original_flat | 0.5 | 48 | 76.067 | 75.996 | 1.579528 |
| 66 | original_flat | 0.6 | 48 | 75.305 | 75.328 | 1.828822 |
| 66 | original_flat | 0.7 | 51 | 68.750 | 68.906 | 2.141490 |
| 66 | head_48 | 0.0 | 34 | 87.348 | 87.278 | 0.983859 |
| 66 | head_48 | 0.1 | 34 | 83.994 | 83.916 | 1.144082 |
| 66 | head_48 | 0.2 | 52 | 82.774 | 82.794 | 1.391515 |
| 66 | head_48 | 0.3 | 60 | 81.555 | 81.559 | 1.322499 |
| 66 | head_48 | 0.4 | 76 | 78.659 | 78.747 | 1.644278 |
| 66 | head_48 | 0.5 | 61 | 76.372 | 76.366 | 1.537700 |
| 66 | head_48 | 0.6 | 60 | 75.915 | 76.061 | 1.879332 |
| 66 | head_48 | 0.7 | 83 | 69.817 | 69.254 | 2.003353 |
| 67 | original_flat | 0.0 | 42 | 87.652 | 87.608 | 0.991583 |
| 67 | original_flat | 0.1 | 67 | 85.061 | 85.003 | 1.088959 |
| 67 | original_flat | 0.2 | 42 | 82.317 | 82.290 | 1.326728 |
| 67 | original_flat | 0.3 | 69 | 80.793 | 80.849 | 1.332738 |
| 67 | original_flat | 0.4 | 122 | 78.049 | 78.140 | 1.622674 |
| 67 | original_flat | 0.5 | 42 | 76.677 | 76.621 | 1.624088 |
| 67 | original_flat | 0.6 | 81 | 74.390 | 74.541 | 1.893869 |
| 67 | original_flat | 0.7 | 63 | 70.884 | 70.167 | 1.872646 |
| 67 | head_48 | 0.0 | 56 | 87.957 | 87.981 | 1.111895 |
| 67 | head_48 | 0.1 | 61 | 84.756 | 84.677 | 1.092135 |
| 67 | head_48 | 0.2 | 73 | 82.317 | 82.395 | 1.448101 |
| 67 | head_48 | 0.3 | 61 | 80.488 | 80.523 | 1.316454 |
| 67 | head_48 | 0.4 | 36 | 78.963 | 78.592 | 1.446614 |
| 67 | head_48 | 0.5 | 54 | 76.220 | 75.726 | 1.525647 |
| 67 | head_48 | 0.6 | 36 | 74.390 | 74.322 | 1.776505 |
| 67 | head_48 | 0.7 | 62 | 68.445 | 68.467 | 2.032045 |
| 68 | original_flat | 0.0 | 45 | 87.348 | 87.178 | 0.971837 |
| 68 | original_flat | 0.1 | 37 | 84.451 | 84.400 | 1.171374 |
| 68 | original_flat | 0.2 | 36 | 83.232 | 83.223 | 1.294181 |
| 68 | original_flat | 0.3 | 42 | 81.250 | 80.890 | 1.334647 |
| 68 | original_flat | 0.4 | 57 | 78.201 | 78.173 | 1.482332 |
| 68 | original_flat | 0.5 | 57 | 76.372 | 76.490 | 1.557666 |
| 68 | original_flat | 0.6 | 36 | 73.933 | 74.004 | 1.844924 |
| 68 | original_flat | 0.7 | 44 | 69.360 | 69.190 | 1.925525 |
| 68 | head_48 | 0.0 | 30 | 86.890 | 86.896 | 0.997144 |
| 68 | head_48 | 0.1 | 30 | 83.994 | 84.019 | 1.163032 |
| 68 | head_48 | 0.2 | 59 | 82.317 | 82.188 | 1.313782 |
| 68 | head_48 | 0.3 | 36 | 81.098 | 81.012 | 1.308893 |
| 68 | head_48 | 0.4 | 29 | 78.811 | 78.861 | 1.512454 |
| 68 | head_48 | 0.5 | 47 | 76.524 | 76.536 | 1.555146 |
| 68 | head_48 | 0.6 | 29 | 73.628 | 73.773 | 1.854563 |
| 68 | head_48 | 0.7 | 47 | 71.037 | 71.088 | 1.910000 |

## Run status

- seed66: {'path': '/data2/yb/remote_experiments/osram_mosi_memory_grid_20261022/execution_v1/runs/head_48', 'reused': True, 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'context_dim': 768, 'flat_adapter': 7894656, 'flat_input_dim': 3328, 'forward_read_dim': 384, 'memory_state_elements_per_conversation': 18432, 'total': 11245857}, 'seconds': 4254.063143517997}
- seed67: {'path': '/data2/yb/remote_experiments/osram_mosi_head48_3seed_20261022/execution_v1/runs/seed_67', 'reused': False, 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'context_dim': 768, 'flat_adapter': 7894656, 'flat_input_dim': 3328, 'forward_read_dim': 384, 'memory_state_elements_per_conversation': 18432, 'total': 11245857}, 'seconds': 1335.8069267820101}
- seed68: {'path': '/data2/yb/remote_experiments/osram_mosi_head48_3seed_20261022/execution_v1/runs/seed_68', 'reused': False, 'status': 'complete', 'epoch': 150, 'error': None, 'parameters': {'context_dim': 768, 'flat_adapter': 7894656, 'flat_input_dim': 3328, 'forward_read_dim': 384, 'memory_state_elements_per_conversation': 18432, 'total': 11245857}, 'seconds': 1303.9387049929937}

Head48 was selected after observing seed66 test results. Candidate-selection and
Test-oracle bias remain; no independent generalization or causal mechanism claim.
Worker source SHA may differ for the seed-routing extension; unchanged numerical core
hashes, same-seed fresh initialization, data/split and exact masks are checked separately.
Original Flat ledgers omitted mixed_rate.py: only the two audited declared Git SHAs
can recover its identical byte hash; actual training mask/update/LR traces are checked by each worker.
Full source/config/initialization provenance is retained in REPORT.json.
