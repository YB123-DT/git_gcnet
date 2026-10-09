# Intermediate status, 2026-10-09 10:14 UTC

INTERNAL DIAGNOSTIC ONLY. Two completed runs (plain384/512), eight live training PIDs. Completed runs both exit0, outputs_verified, all40 artifact hashes verified. No training/inference/configuration changes made during inspection.

Per-rate best Test-oracle W-F1 (%); incomplete runs have unequal budgets and are not final comparisons.

| Width | Plain epochs | Plain mean8 | Plain high | Nested epochs | Nested mean8 | Nested high |
|---|---:|---:|---:|---:|---:|---:|
|384|100|80.137954|75.107050|93|80.252511|75.632588|
|512|100|78.991952|74.197002|90|80.122986|75.569412|
|768|98|79.846990|74.875282|77|79.717799|74.846370|
|1024|83|80.141946|75.925914|67|78.702930|73.636372|
|1280|75|79.016399|73.817337|71|79.420213|74.682411|

Notably768/1280 Nested recovered substantially from previous intermediate71.429/53.539, so those partial low scores were not final outcomes. Current best new Nested combination is384 at80.252511/75.632588, still below the completed width256+Nested80.362122/75.899420 reference. Do not decide the preferred width until all ten100epoch runs complete.

Sources: remote `/data2/yb/remote_experiments/osram_medium_flat_nested_20261009/runs/*/{DISPATCH.json,seed_66/history.json,seed_66/metrics.json}`. Completed metrics used when present; otherwise earliest per-rate maxima over available history. Same source910a3dc, GPU2 plain/GPU3 Nested.
