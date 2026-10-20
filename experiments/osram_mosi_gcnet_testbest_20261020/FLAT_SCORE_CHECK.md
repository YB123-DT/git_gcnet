# Flat score check,2026-10-20

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. Read-only analysis; no new training or inference.

Remote summaries under `execution_v2/runs/MOSI/seed_{66,67,68}/flat` all
confirm complete150epochs,GCNet-random-mask-v1,per-rateTest-best weighted_f1.
Sourcefd96f540e807426d6b4d858febdc2715b3a79a68; originalMOSI s0 environment.

| Seed | Mean8 W-F1 | High .5/.6/.7 W-F1 |
|---:|---:|---:|
|66|79.12802655317742|73.33810876241544|
|67|78.75085031827453|72.65892008810142|
|68|78.79388048154536|72.72661764342057|
|Mean|78.89091911766577|72.90788216464581|

All150training parity entries for eachseed pass exact actual training masks,
optimizer step counts and learningrates against the precedingGCNet validation
round; maximum observed trainMSE difference0.0 in allthree histories. Only the
selection rule differs between these two rounds. Previous validation-selected
Flat mean8 was76.4216: restoringTest-best changes the reported score, not the
trained trajectories. No claim of parity against the older80.747round.

Older80.747 mean from `../osram_complete150_20261011/RESULT.md` used
conversation-mask-v2. Currentdifference is-1.856081pp (oldreference rounded).
Both use per-rateTest-best and150epochs, but the missing samplers differ in
training/evaluation and consume randomness differently. In particular original
GCNet rate.7 enforces exactly one observed modality, while independent
Bernoulli-plus-nonempty-repair can retain pairs/triples at.7. Same nominal rate
does not imply the same distribution or same utterance masks.

Code evidence:`gcnet_modality_jepa/gcnet_legacy_random_mask.py::random_mask`,
`gcnet_modality_jepa/mask_schedule.py::ConversationMaskSchedule.generate`,
and the legacy/current configs plus strict oldcheckpoint prediction bridge in
`../osram_mosi_gcnet_router_3seed_20261019/worker.py`.

Do not attribute the entire old/newscore gap causally to the sampler from this
comparison alone; a matchedfactor analysis has not been performed. This score
is for Flat,so it cannot be blamed on addingRouter. Do not change the requested
GCNet missing protocol in response to the lower score.

Separately,the contemporaneous IEMOCAP-6 numbers around54–56 were fold1
intermediate training scores,not completedfivefoldmeans. Do not conflate them
with this completedMOSI score.
