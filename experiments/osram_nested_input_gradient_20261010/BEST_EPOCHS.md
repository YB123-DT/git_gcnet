# Original Flat versus original Nested selected epochs

INTERNAL DIAGNOSTIC ONLY. Read from remote metrics.json on 2026-10-10.
These are eight per-rate Test-oracle selected epochs per training run, not a
single global best epoch. Both methods train once per seed for 100 epochs.
This comparison excludes the newer Gap Gate variants. No training or inference.

| Missing rate | Flat66 | Nested66 | Flat67 | Nested67 | Flat68 | Nested68 |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 76 | 91 | 69 | 55 | 44 | 73 |
| 0.1 | 76 | 84 | 70 | 47 | 54 | 73 |
| 0.2 | 76 | 84 | 70 | 47 | 48 | 92 |
| 0.3 | 76 | 94 | 70 | 47 | 88 | 83 |
| 0.4 | 76 | 71 | 72 | 47 | 48 | 94 |
| 0.5 | 48 | 62 | 70 | 47 | 88 | 76 |
| 0.6 | 76 | 71 | 87 | 48 | 87 | 73 |
| 0.7 | 39 | 98 | 72 | 49 | 57 | 88 |

Sources on biggpu:

- Flat: `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_{66,67,68}/metrics.json`.
- Nested66: `/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66/metrics.json`.
- Nested67/68: `/data2/yb/remote_experiments/osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence/seed_{67,68}/metrics.json`.

All six files report `selection_protocol=per-rate-test-oracle`,
`best_epoch=null`, and the table above in `selected_epoch_by_rate`.
Nested peaks later for seeds66/68, earlier for seed67; no uniform delay is shown.
