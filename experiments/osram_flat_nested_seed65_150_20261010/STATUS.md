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
Remote root and commands in LAUNCH.md. No restart or protocol changes made at
that observation.

## 2026-10-10 11:23 UTC recovery

Both original PIDs disappeared and the default tmux server/sessions vanished.
Logs ended without Python traceback; provenance still said running. Host uptime
19days rules out a reboot; memory currently ample, but kernel log access denied.
Termination cause is unresolved; do NOT assert OOM or code failure.

History/log had reached Flat82/Nested60, but last FULL committed states contain
Flat81 (Adam step162) and Nested59 (step118). Restore from these committed states,
not the uncommitted history tail or BEST weights. Both contain model/optimizer,
RNG/schedule/selection and8 checked BEST references. Source and observer hashes
match archived identities. Original checkpoint hashes before recovery:

- Flat:9f274d90a6e9c383ac655c02eaf9120bcb76d1b12874534df27dc0ade1522416
- Nested:715f924f1f733b664bf45779a0222fef929053bf2098b3fd6341db26187a63dd

Preserved logs as logs/{flat,nested}_100_interrupted.log. Restarted SAME commands
from LAUNCH.md with log append, no source/config changes, now in dedicated tmux
socket `gcnet_seed65` to isolate from default tmux server lifecycle. Sessions
unchanged names; Flat PID3013498/Nested PID3013574. Both live and provenance
resumed=true. Subsequent progress still requires live committed-history checks.
Stage2 remains conditional on successful100 completion. No training from scratch.

Post-recovery verification: Flat committed history84, Nested61. Replayed Flat82
and Nested60 printed train/test/loss values match interrupted logs at logged
precision. Both advanced past their interruption points, confirming actual
training restoration rather than just live initialization processes.

## Next live check: Flat118 / Nested87

Flat original100 completed with verified outputs:8-rate79.931940%, high74.941902%.
Its continuation is live, saved history118,19 LR traces (epoch119 already entered).
Cumulative provisional8-rate79.951281%, high74.983571%; new-only101–118
per-rate maxima mean79.400361%, high74.526179%. Do not claim final150 performance.

Nested original100 is live with87 saved epochs: provisional8-rate79.715014%,
high74.855673%. Its continuation has not started. Both current PIDs alive; no new
failure recorded. Budgets remain unequal; final paired conclusion pending.

## Next live check: Flat142 / Nested102

Both original100 stages complete with verified outputs. Original1008-rate/high:
Flat79.931940/74.941902%, Nested79.768669/74.867331%.
Both continuation PIDs alive. Flat saved142epochs (43 LR traces, epoch143 entered),
Nested102 (3 LR traces, epoch103 entered). Cumulative provisional8-rate/high:
Flat80.009566/74.983571%, Nested79.805028/74.867331%.
New-only per-rate maxima: Flat79.564152/74.611626%, Nested77.834254/72.071762%.
Still unfinished and unequal budgets; no final150 comparison or failure claim.
