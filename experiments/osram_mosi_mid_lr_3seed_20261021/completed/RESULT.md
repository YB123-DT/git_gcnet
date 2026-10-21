# MOSI Flat3e-4→3e-5 three-seed replication

INTERNAL TEST-ORACLE SCREEN

Seed66 is reused read-only;only67/68 newly trained.150epochs,original GCNet missing.
Per-rate TEST W-F1-best checkpoints,not a single deployment model. No new block or loss.

| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|
| 66 | original | 79.1159 | 79.1280 | 73.2724 | 73.3381 |
| 66 | lr3e4 | 79.2683 | 79.2491 | 73.3740 | 73.4098 |
| 67 | original | 78.7348 | 78.7509 | 72.6118 | 72.6589 |
| 67 | lr3e4 | 79.4779 | 79.4024 | 73.9837 | 73.7766 |
| 68 | original | 78.8872 | 78.7939 | 72.9167 | 72.7266 |
| 68 | lr3e4 | 79.2683 | 79.1937 | 73.2215 | 73.2281 |

| Model | Mean ACC ± SD | Mean W-F1 ± SD | High ACC ± SD | High W-F1 ± SD |
|---|---:|---:|---:|---:|
| original | 78.9126 ± 0.1918 | 78.8909 ± 0.2065 | 72.9336 ± 0.3306 | 72.9079 ± 0.3741 |
| lr3e4 | 79.3382 ± 0.1210 | 79.2817 ± 0.1081 | 73.5264 ± 0.4033 | 73.4715 ± 0.2794 |

Per-seed/per-rate scores and selected epochs are in METRICS.csv. Standard deviations are sample SD across three seeds.
