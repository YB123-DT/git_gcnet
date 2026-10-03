# D3-W256 three-seed confirmation — complete

Seed66 reused;67/68 completed100epochs and each retained8best checkpoints.
INTERNAL per-rate Test-oracle diagnostics, not formal validation-selected paper results.
W-F1%, differences percentage points. No last-epoch substitution.

|Seed|Flat eight-rate|D3-W256 eight-rate|Delta|Flat high|D3-W256 high|Delta|
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.932|-0.136|76.352|76.200|-0.152|
|67|80.556|79.583|-0.973|75.990|74.801|-1.188|
|68|80.053|79.345|-0.708|74.440|73.975|-0.464|
|Mean|80.559|79.953|-0.606|75.594|74.992|-0.602|

|Model|Eight-rate mean ± sample SD|High missing mean ± sample SD|
|---|---:|---:|
|Flat|80.559 ± 0.507|75.594 ± 1.016|
|D3-W256|79.953 ± 0.856|74.992 ± 1.125|

Both eight-rate and high-missing scores are lower than Flat for all three seeds.
Seed66 near-parity does not persist as a robust result. Selected after four seed66 configurations;
no significance claim, new tuning, or new training in this report.
