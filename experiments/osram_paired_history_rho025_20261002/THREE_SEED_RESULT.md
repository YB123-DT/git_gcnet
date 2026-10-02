# Three rho values, three seeds — complete

Seed66 reused; six new seed67/68 runs completed100epochs. Original per-rate BEST, not last epoch.
All nine per-rate summaries verified; per-seed masks identical across rho values.
INTERNAL TEST-ORACLE; candidates selected from a seed66 sweep, not a formal validation protocol.
Mean ± sample standard deviation across seeds66/67/68. Units W-F1%, deltas percentage points.

|Model|8-rate mean ± SD|Delta|High missing mean ± SD|Delta|
|---|---:|---:|---:|---:|
|Flat|80.559 ± 0.507|—|75.594 ± 1.016|—|
|0.05|79.927 ± 0.630|-0.632|75.315 ± 0.859|-0.279|
|0.10|80.260 ± 0.883|-0.299|75.297 ± 1.114|-0.297|
|0.25|80.014 ± 0.446|-0.545|75.274 ± 0.836|-0.320|

|rho|seed|8-rate|Delta vs seed-matched Flat|High missing|Delta vs Flat|
|---|---:|---:|---:|---:|---:|
|0.05|66|80.551|-0.517|76.130|-0.222|
|0.05|67|79.936|-0.620|75.396|-0.594|
|0.05|68|79.292|-0.761|74.418|-0.022|
|0.10|66|81.231|+0.163|76.203|-0.149|
|0.10|67|80.045|-0.511|75.635|-0.355|
|0.10|68|79.504|-0.549|74.053|-0.387|
|0.25|66|80.419|-0.649|75.460|-0.893|
|0.25|67|80.087|-0.469|76.001|+0.012|
|0.25|68|79.537|-0.517|74.361|-0.079|

All three rho values have lower mean eight-rate and high-missing W-F1 than Flat.
The positive seed66 eight-rate result at rho.10 did not reproduce in seeds67/68.
No significance claim; no new tuning, inference, or fixed-pattern experiments in this report.
