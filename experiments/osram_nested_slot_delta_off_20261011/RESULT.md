# Nested per-slot correction OFF

INTERNAL DIAGNOSTIC ONLY

MOSI seed66, original 100epoch Nested per-rate Test-oracle BEST. Frozen
evaluation only; no training. Memory computed once per batch, identical
weights/query/write/masks/Local Skip/Flat/head. Remove a correction by
restoring its raw evidence, not by zeroing the evidence itself.

Source code e31f4a2; server biggpu GPU7. Eight checkpoint/model hashes
unchanged, labels/masks equal original archive, predictions agree <=1e-5.

## Overall W-F1 (%)

| Disabled delta | Mean8 | High (.5/.6/.7) | OFF-minus-ON pp | Corrections / harms (OFF -> ON) |
|---|---:|---:|---:|---:|
|None (Full)|80.992147|76.077229|0|—|
|Local|81.030615|76.179811|+0.038468|0 / 2|
|Base|80.972037|76.005851|-0.020110|12 / 10|
|Gap-A|80.974021|76.028894|-0.018126|1 / 0|
|Gap-T|80.950288|76.007601|-0.041858|11 / 8|
|Gap-V|80.980083|75.935049|-0.012064|5 / 4|

## Current observed modality groups

Equal mean over nonempty rate groups, NOT whole-conversation fixed missing.
T is Text. Group N and counts available in per_rate.csv. Small low-rate
groups can make macro differences noisy; repeated samples are not unique.

| Observed | Full | Local OFF | Base OFF | Gap-A OFF | Gap-T OFF | Gap-V OFF |
|---|---:|---:|---:|---:|---:|---:|
|A|73.405607|73.505919|73.267021|73.405607|72.998760|73.401970|
|T|81.252728|81.252728|81.149553|81.149553|81.252728|81.149553|
|V|63.111058|63.111058|63.252826|63.111058|63.252826|63.111058|
|AT|85.484095|85.484095|85.675256|85.484095|85.484095|85.675256|
|AV|66.708644|66.933853|66.912849|66.708644|66.556765|66.708644|
|TV|89.010378|89.010378|89.010378|89.010378|89.010378|89.010378|
|ATV|86.796521|86.796521|86.771403|86.796521|86.796521|86.796521|
|no_Text|68.393719|68.486782|68.427627|68.393719|68.316910|68.371230|

## Interpretation and limits

Each individual correction has a small total effect at this trained working
point. Gap-T has 11 corrections / 8 harms; Local has 0 / 2. This is direct
inference contribution, not separately retrained ablation. Other corrections
remain ON, so effects are conditional and need not sum to all-delta-off.
It does not prove no useful information exists or that training had no effect.
Inactive Gap remains masked. No automatic extra seeds/training.

Artifacts: results/{STATUS.json,SUMMARY.json,utterances.csv,per_rate.csv,macro.csv}.
Implementation: prior runner --per-slot and readout.disable_correction_slot.
Launch command and PID in RUNNING.md; checkpoints remain remote.
