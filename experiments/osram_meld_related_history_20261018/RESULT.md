# MELD B1/B2 Local-history residual

INTERNAL TEST-ORACLE SCREEN

Seed66,50epochs,Adam lr1e-5 wd2e-4; ACC-best same-forward UA/W-F1.
Original O6 Flat reused. All full forward Base/Gap slots preserved.

Status: both50epochs complete after restoring full epoch46 states following
the host reboot. Same immutable442596a source, original initialization and
all50 train/selection mask streams verified. No new inference/training is
performed to obtain this report. Added parameters: B1682,834; B2766,163.
All per-rate UA/W-F1 are from the same ACC-selected epoch and saved masks.
Matching mask streams do not imply that differently selected epochs have
identical selected masks; B2 rate0.1 selects epoch10 vs Flat/B1 epoch13.

| Model | ACC | UA | W-F1 | high ACC | high UA |
|---|---:|---:|---:|---:|---:|
| Flat | 61.786 | 38.435 | 58.918 | 57.880 | 32.411 |
| B1 | 61.796 | 38.502 | 58.936 | 57.867 | 32.460 |
| B2 | 61.758 | 38.005 | 58.809 | 57.842 | 32.388 |

B1 minus Flat: ACC+0.010pp,UA+0.067pp,W-F1+0.018pp — effectively unchanged
at this single-seed screening resolution, not evidence of a reliable gain.
B2 minus Flat: ACC−0.029pp,UA−0.430pp,W-F1−0.109pp. No automatic expansion.
These comparisons do not establish the cause of B2's decline or a causal
interpretation of predicted consistency. No statistical significance claim.

| Model | rate | epoch | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|
| Flat | 0.0 | 8 | 67.663 | 44.495 | 65.861 |
| Flat | 0.1 | 13 | 65.556 | 45.739 | 64.316 |
| Flat | 0.2 | 11 | 64.176 | 40.984 | 61.960 |
| Flat | 0.3 | 21 | 62.414 | 40.244 | 60.124 |
| Flat | 0.4 | 35 | 60.843 | 38.784 | 58.146 |
| Flat | 0.5 | 13 | 59.732 | 33.210 | 55.657 |
| Flat | 0.6 | 24 | 57.433 | 32.168 | 53.087 |
| Flat | 0.7 | 46 | 56.475 | 31.856 | 52.191 |
| B1 | 0.0 | 8 | 67.663 | 44.438 | 65.844 |
| B1 | 0.1 | 13 | 65.556 | 45.739 | 64.312 |
| B1 | 0.2 | 11 | 64.100 | 40.931 | 61.879 |
| B1 | 0.3 | 21 | 62.490 | 40.541 | 60.226 |
| B1 | 0.4 | 35 | 60.958 | 38.984 | 58.303 |
| B1 | 0.5 | 13 | 59.693 | 33.198 | 55.623 |
| B1 | 0.6 | 24 | 57.433 | 32.216 | 53.122 |
| B1 | 0.7 | 46 | 56.475 | 31.965 | 52.179 |
| B2 | 0.0 | 8 | 67.625 | 44.460 | 65.824 |
| B2 | 0.1 | 10 | 65.594 | 42.433 | 63.689 |
| B2 | 0.2 | 11 | 64.061 | 40.949 | 61.873 |
| B2 | 0.3 | 21 | 62.375 | 40.458 | 60.099 |
| B2 | 0.4 | 35 | 60.881 | 38.576 | 58.158 |
| B2 | 0.5 | 13 | 59.732 | 33.210 | 55.656 |
| B2 | 0.6 | 24 | 57.318 | 32.152 | 53.018 |
| B2 | 0.7 | 46 | 56.475 | 31.802 | 52.151 |
