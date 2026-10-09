# Intermediate status, 2026-10-09 09:59 UTC

INTERNAL DIAGNOSTIC ONLY. All10 training PIDs verified alive. No runs complete; no new training/inference or configuration changes during inspection.
Values are per-rate best Test-oracle W-F1 so far, percentages. Different completed epochs make these provisional, not equal-budget final comparisons.

| Flat width | Plain epochs | Plain mean8 | Plain high | Nested epochs | Nested mean8 | Nested high |
|---|---:|---:|---:|---:|---:|---:|
|384|77|79.830733|74.969048|58|78.662828|73.486219|
|512|66|78.117596|72.983133|57|78.522976|73.105368|
|768|58|79.260703|74.314405|50|71.429391|64.970095|
|1024|49|77.816345|72.828383|40|76.552164|72.346944|
|1280|42|75.910249|71.902210|51|53.538734|54.147731|

The 768/1280 Nested combinations currently perform much worse, particularly1280. The inspected last three log lines contain finite losses and continuing epochs; this does not establish a root cause or rule out implementation problems. Do not interpret low intermediate scores as evidence that more parameters necessarily hurt; wait for the fixed100epoch budget, then evaluate all runs without hiding failures.
Existing completed width256+original Nested reference:80.362122 mean8/75.899420 high. This remains above all currently observed Nested combinations, but final comparison is pending.

Source: remote `runs/*/seed_66/history.json`, per-rate max over completed epochs; process status from corresponding `DISPATCH.json` plus PID existence. Server biggpu, GPU2 plain/GPU3 Nested, sealed source910a3dc.
