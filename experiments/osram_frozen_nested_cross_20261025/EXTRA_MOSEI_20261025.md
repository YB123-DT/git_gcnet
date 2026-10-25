# User-authorized extra MOSEI admission

User requested MOSEI priority, then explicitly clarified: preserve all running
experiments and add the outstanding MOSEI jobs immediately. Do not migrate,
restart, pause or terminate training workers.

At execution,rate0.5 had already started automatically (PID1003944). Only two
still-pending jobs were admitted, with original unchanged commands/config/data/
source/masks and output directories:

| Rate | Physical GPU | PID |
|---|---:|---:|
|0.6|0|1010894|
|0.7|1|1010895|

The extra capacity uses measured peaks and live free VRAM/RAM, allowing at most
one temporary extra worker per card (peak5instead of4). Normal automatic queue
admission remains at4/card; it will not add more workers while over that ceiling.
The finite training count remains56,not58. All eight MOSEI rates are now active.

## Safe queue coordination

The one-time operational helper is committed as4dd2e87 and deployed separately
under `controls_4dd2e87`; no immutable training snapshot was edited. It validates
the exact controller/root/source/UID/cwd,quiesces only the queue controller,
rejects unpersisted children,checks capacity,and acquires scheduler.lock before
queue writes/replacement. It registers the two new PIDs using the original
launch function and resumes the original immutable controller with`--resume`,
so these jobs cannot later be launched twice. Controller946470was replaced by
1010896; no signal was sent to any training worker or process group.

The live receipt `EXTRA_MOSEI_20261025.json` records old/new controller identity,
helperSHA256,full commands/UUIDs,the12preserved worker PIDs and new admissions.
All12were verified still running or naturally complete after handoff. Focused
tests3/3passed,including lock-timeout refusal to overwrite queue/start a second
controller; code review accepted. Tests/operation do not establish model gains.

Verified2026-10-25T23:54:02UTC:queue train,14running/29complete/13pending,no
reported errors. MOSEI rates0.0..0.7epochs49/37/36/26/25/5/0/0respectively;
the last two had completed initialization and begun formal training. Exact live
worker commands matched queue records. Source remains
`2e3e710e54cdd85f9e660470a7d1ef1b3f93462e`,cmerc-repro,newGCNetMissing.
