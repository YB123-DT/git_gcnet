# Status 2026-10-09 10:17 UTC

INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle best so far. Four complete, six live. Newly completed flat384_nested and flat768: exit0, outputs_verified, all40 artifact hashes verified. Earlier completed plain384/512 verified previously. No additional training/inference launched.

Adapter dimensions: input4352 = Local256 + Base1024 + three Gap1024 slots; hiddenW is varied; output1600 unchanged. Thus384 denotes4352->384->1600, not a change to OSRAM head count or output size. Forward-only history occupies forward512 in each1024 context slot; original Flat interface remains unchanged.

| Width | Plain epoch | Plain mean8 | Plain high | Nested epoch | Nested mean8 | Nested high |
|---|---:|---:|---:|---:|---:|---:|
|384|100|80.137954|75.107050|100|80.252511|75.632588|
|512|100|78.991952|74.197002|98|80.284120|75.569412|
|768|100|79.846990|74.875282|85|79.739420|74.846370|
|1024|93|80.310172|76.022643|74|78.726311|73.698721|
|1280|82|79.269375|74.388071|76|79.938430|75.081056|

WF1 in percent.512+Nested best so far now80.284 versus completed256+Nested80.362, but final results pending.1280+Nested continues recovering. Do not infer final width ranking from unequal epochs.
Source remote osram_medium_flat_nested_20261009/runs/*/seed_66 metrics/history and DISPATCH statuses. Six running PIDs verified alive.
