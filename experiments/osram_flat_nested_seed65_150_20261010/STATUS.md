# Seed65 interim status

INTERNAL DIAGNOSTIC ONLY. Observed2026-10-10 11:16 UTC on biggpu GPU7.
Both original100 stage processes are running and alive; neither continuation50
has started yet. Pipeline will start continuation only after successful100 exit.

|Model|Completed epochs / planned total|Provisional8-rate BEST W-F1%|Provisional high%|
|---|---|---:|---:|
|Flat|71/150|79.133063|73.887744|
|Nested|53/150|75.668180|70.746203|

Scores computed from saved history: independent per-rate maximum weighted_f1
over completed epochs, then mean across8 rates or0.5/0.6/0.7. Final metrics.json
is not yet present. These are interim Test-oracle maxima at unequal epoch
budgets, NOT final scores, same-epoch comparisons, or evidence of a final failure.
Remote root and commands in LAUNCH.md. No restart or protocol changes made.
