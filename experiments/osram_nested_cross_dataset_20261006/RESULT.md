# Old Nested cross-dataset screening

INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.

## Historical six-class Flat numerical comparison

Same seed66 and five held-out sessions; historical metrics archived under
raw/Flat_IEMOCAPSix_seed66. Historical per-rate Test-oracle metrics omit
selection_metric, whereas current Nested explicitly selects by accuracy.
This is a numerical comparison, NOT a proven identical-selection ablation.
Do not attribute the entire difference to the Nested module alone.

|Summary (%)|Historical Flat|Nested|Nested minus Flat (pp)|
|---|---:|---:|---:|
|8-rate W-F1|61.104636|60.431700|-0.672935|
|High-missing W-F1|59.001275|58.227977|-0.773298|
|8-rate Accuracy|61.377635|60.878719|-0.498916|

## ACC and UA audit from existing predictions

No new training/inference. `acc_ua.py` reads80 existing prediction files
(2models x5folds x8rates), verifies equal labels/availability order and
recomputed ACC equals each selected metrics.json Accuracy. UA is mean recall
across all six classes; class supports and prediction hashes saved in
ACC_UA_SUMMARY.json. Two unit tests distinguish UA from imbalanced ACC and
reject absent/invalid classes. All actual folds include all six classes.

|Summary (%)|Historical Flat|Nested|Nested minus Flat (pp)|
|---|---:|---:|---:|
|8-rate ACC|61.377635|60.878719|-0.498916|
|8-rate UA|60.267962|59.402902|-0.865060|
|High-missing ACC|59.243860|58.776130|-0.467730|
|High-missing UA|57.967122|57.339395|-0.627727|

Five-fold equal-weight aggregation, not pooled prediction metrics. UA is
NOT macro F1. Both metrics use the same selected checkpoints as the W-F1
report; historical Flat selection_metric remains unrecorded, so the same
non-identical-selection caveat applies. INTERNAL DIAGNOSTIC ONLY.

## Authorized three-seed extension: Four, Six, MOSEI

User explicitly requests all three datasets and permits GPU3 in addition
to GPU6. Finite27 NEW runs: IEMOCAPFour seeds66/67/68 x5folds (15),
IEMOCAPSix seeds67/68 x5folds (10), MOSEI seeds67/68 x1split (2).
Existing Six seed66 fivefolds and ongoing MOSEI seed66 are reused, never
dispatched again. Total requested completed sets:15Four+15Six+3MOSEI.

Model is unchanged old Nested, not Local8/root-aware. Each run copies exact
dataset/class/seed/fold Flat config and changes only the Nested switch.
Four-class label/split pickle is separately hashed; actual same feature roots
are reused.100epochs, random cyclic0.0–0.7, task-only loss and original
optimizer/lr/batch retained. IEMOCAP selects per-rate Accuracy, MOSEI W-F1;
all selected checkpoints/predictions and recovery states retained.

New INPUTS_3SEED.json pins all27 references and data without overwriting
INPUTS.json or seed66 archives. Multi-seed changes only runner paths/dispatch;
old immutable source33ded09 keeps serving the currently running seed66 run.
GPU whitelist=[3,6], maximum3 NEW jobs per card,8000MiB admission reservation
per job (including CPU-loading children whose VRAM is not yet visible).
Live free GPU3~32GiB, GPU6~14GiB at initial check; recheck each launch.
No use of GPU4, no killing unrelated jobs, no protocol alteration to fit VRAM.
Failure halts new admissions and drains current jobs; no blind retry.

Five runner tests pass: legacy seed66 task scope/only-switch behavior,
invalid protocols, exactly27 new tasks without duplicates, class/seed-specific
reference and output paths. Existing model checks reused; no new GPU smoke.

Commands from new sealed source:
```text
python -m experiments.osram_nested_cross_dataset_20261006.multi_seed --prepare
python -u -m experiments.osram_nested_cross_dataset_20261006.multi_seed
```
Persistent queue: QUEUE_3SEED.json. Deployment verified: source644ee14
sealed/pushed; INPUTS_3SEED.json created, all27 exact-reference configs and
four-/six-class label hashes checked. Remote five runner tests passed.
Coordinator PID137772, tmux nested_three_seed_20261006. Initial launch snapshot:
MOSEI seed67 PID137890 GPU3; MOSEI seed68 PID137900 GPU6;25pending.
LAUNCH_3SEED.json records exact commands/UUIDs and full27-task scope.
GPU3/6 remaining capacity automatically admits pending Four/Six folds;
seed66 MOSEI old source and completed Six seed66 remain untouched.
Status: running/initial data loading. No new three-seed aggregate claimed.

## MOSEI three seeds complete

All three MOSEI seeds66/67/68 complete100epochs with verified outputs.
Per-rate BEST Test-oracle weighted F1, not final-epoch/single-epoch average:

|Seed|8-rate W-F1 (%)|High-missing W-F1 (%)|
|---|---:|---:|
|66|83.880953|81.322609|
|67|83.876906|81.732793|
|68|83.772477|81.566440|
|Mean|83.843445|81.540614|

At this status check, additional27-task queue has20complete/4running/3pending,
no reported error. IEMOCAP three-seed full fivefold summaries remain pending.
All values INTERNAL DIAGNOSTIC ONLY. No statistical improvement claim.

## FINAL: all three datasets / three seeds complete

Additional queue27/27 complete, no reported error. Including reused Six
seed66 and MOSEI seed66, all33 requested runs are complete: Four15folds,
Six15folds,MOSEI3splits. Each run100epochs, expected class/dataset/seed/fold
and old-Nested switch verified. Eight BEST paths and full recovery checkpoint
present per run. All264 saved prediction hashes match recorded provenance;
IEMOCAP ACC recomputed from predictions agrees with selected metrics.
No new inference/training. Current partial-snapshot report is superseded.

Aggregation: equal folds within seed, equal eight rates, equal three seeds.
SD is sample standard deviation across THREE seed summaries (not15folds).
UA is class recall mean, not macro-F1; no UA reported for MOSEI regression.
Use per-rate BEST (IEMOCAP Accuracy selection; MOSEI W-F1 selection), not
last epoch or a single shared-best epoch. INTERNAL DIAGNOSTIC ONLY.

|Dataset|8-rate W-F1 (%)|8-rate ACC (%)|8-rate UA (%)|
|---|---:|---:|---:|
|IEMOCAPFour|78.933 ±0.474|79.054 ±0.412|79.477 ±0.399|
|IEMOCAPSix|61.069 ±0.562|61.469 ±0.512|60.119 ±0.621|
|CMUMOSEI|83.843 ±0.061|83.995 ±0.090|not reported|

|Dataset|High-missing W-F1 (%)|High-missing ACC (%)|High-missing UA (%)|
|---|---:|---:|---:|
|IEMOCAPFour|76.863 ±0.484|77.002 ±0.400|77.355 ±0.458|
|IEMOCAPSix|58.822 ±0.515|59.316 ±0.474|57.890 ±0.491|
|CMUMOSEI|81.541 ±0.206|81.777 ±0.258|not reported|

|Dataset|Seed|8-rate W-F1 (%)|8-rate ACC (%)|
|---|---:|---:|---:|
|Four|66|79.384|79.453|
|Four|67|78.438|78.631|
|Four|68|78.976|79.078|
|Six|66|60.432|60.879|
|Six|67|61.493|61.793|
|Six|68|61.284|61.735|
|MOSEI|66|83.881|83.988|
|MOSEI|67|83.877|84.088|
|MOSEI|68|83.772|83.909|

THREE_SEED_SUMMARY.json stores full per-seed/per-fold/per-rate scores and
provenance/metrics/history hashes. COMPLETION_3SEED.json records terminal
queue. Raw metrics/config/provenance archived for all33 runs. Full histories,
checkpoints and predictions remain on biggpu. summarize_three_seed.py
reproduces the audit using saved artifacts. Training commits33ded09/644ee14
differ in launch/report code only, not model/trainer modules.
No formal validation-selected/generalization or statistical improvement claim.
Status: this finite three-dataset/three-seed experiment batch COMPLETE.

User requested Nested on MOSEI and IEMOCAP. Use the original
`nested_gnn_rooted_evidence` (Local single node, eight Memory head nodes per
active evidence, 64d, three GIN layers), not Local8/root-aware variants.
No changes to model code, Memory read/write/query, original task head/loss,
random cyclic missing protocol, optimizer or dataset-specific preprocessing.

## Fixed scope and protocol

- MOSEI: seed66, official single split,100epochs, missing rates0.0–0.7.
- IEMOCAP: six classes, seed66, five leave-one-session-out folds,100epochs
  each, same eight missing rates. No three-seed expansion.
- New configs copy each exact dataset/fold Flat config; only
  `osram_meaningful_block` changes. Null rate list in historical config uses
  the trainer's existing default eight rates, not a new sampling schedule.
- Per-rate BEST Test-oracle screening. Current trainer selects MOSEI by
  weighted F1 and IEMOCAP by accuracy; report IEMOCAP W-F1 and accuracy.
- Existing Flat metrics do not record selection_metric. Do NOT label them
  as proven identical checkpoint-selection controls. Reuse them for masks,
  inputs and protocol audit, and flag any historical numerical comparison.
- Compare ordered and canonical evaluation masks on completion; retain all
  eight BEST checkpoints, predictions and complete last_training.pt state.
- Server biggpu. GPU5 dedicated to MOSEI; GPU6 runs IEMOCAP folds sequentially.
  GPU4 forbidden. Capacity checks do not alter batch size or training protocol.
- Source pinned via git archive and SNAPSHOT.json; INPUTS.json hashes feature
  files, label/split pickle, exact reference configs and metrics.
- Failed child stops that lane; no blind rerun or silent weights-only resume.

## Commands

From sealed source, using `/data2/yb/reproduction_workspace/envs/s0/bin/python`:

```text
python -m experiments.osram_nested_cross_dataset_20261006.run --prepare
python -u -m experiments.osram_nested_cross_dataset_20261006.run --lane mosei --gpu 5
python -u -m experiments.osram_nested_cross_dataset_20261006.run --lane iemocap --gpu 6
```

Root: `/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006`.
Persistent tmux queues record PID, command, UUID, log and output paths.

## Verification and status

Three focused runner tests: exact six-task scope, only old-Nested config
switch changes, invalid protocol/fold/rates/interventions rejected.
Tests first failed on absent runner, then passed. No repeated model smoke:
the unchanged old Nested model already has regression-test evidence.

## Launch snapshot

Source commit33ded09 (pushed), sealed archive:
`/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/source_33ded09`.
INPUTS.json prepared successfully with both datasets and six reference folds.
Remote s0 environment also passed all three runner tests.
Started2026-10-06T01:31:36 UTC:

| Lane | Host GPU | Queue PID | First child PID | tmux session |
| --- | ---: | ---: | ---: | --- |
|MOSEI|5|3001323|3001353|nested_mosei_20261006|
|IEMOCAP six-class|6|3001326|3001355|nested_iemocap_20261006|

LAUNCH_MOSEI.json and LAUNCH_IEMOCAP.json preserve exact commands/UUIDs/paths.
Live checks confirm both child PIDs; IEMOCAP logs show feature loading.
This records successful process launch, not epoch completion or final scores.
IEMOCAP remaining folds are queued in the same persistent process.

Status: running (initial data loading). No new scores available.

## GPU6 parallel scheduling update

User authorized placing remaining IEMOCAP folds on GPU6 concurrently.
At resource check: GPU6 free27231MiB; active fold3 used5260MiB; /data2 free249GiB.
fold1/2 completed; fold3 remains untouched. Transfer undispatched folds only.
`parallel_remaining.py` verifies and suspends only the old coordinator, rereads
its queue to avoid races, starts fresh pending folds from source_33ded09,
and monitors all folds. The paused coordinator is retired only after its
existing training child finishes. No training child is signalled/restarted.
Admission rechecks GPU6 UUID/free memory before each launch. Model, data,
seed,100epoch budget, batch size, losses and checkpoint policy are unchanged.
Two scheduling tests passed after failing on the absent module; no GPU smoke.
Failed handoff before dispatch resumes original coordinator; after dispatch
requires reconciliation rather than silently creating duplicate jobs.

Parallel launch verified: coordinator3554889, new fold4 PID3554897,
new fold5 PID3555082. Original fold3 PID3394998 remains running. Old queue
coordinator3001326 is stopped (T), not its training child. Exact transferred
folds=[4,5]; LAUNCH_PARALLEL_IEMOCAP.json preserves commands and source hash.
Coordinator code5503065 pushed; all training continues from33ded09.
No new scores claimed; GPU5 MOSEI left untouched.

## Completed IEMOCAP six-class results

All five folds seed66 completed100epochs; all PROVENANCE records report
complete and outputs_verified=True. Eight BEST checkpoint paths per fold
are present. Metrics/history/config/provenance archived under raw/.
This experiment DID NOT train IEMOCAPFour. Historical Flat four-class
outputs exist separately and are not Nested four-class results.

Per-rate Test-oracle selected by Accuracy, then report W-F1 at those SAME
checkpoints. Five folds equal-weight mean, then rates equal-weight mean;
not pooled out-of-fold F1, not a three-seed result, not final-epoch metrics.

|Missing rate|W-F1 (%)|Accuracy (%)|
|---|---:|---:|
|0.0|63.389|63.751|
|0.1|62.458|62.902|
|0.2|61.069|61.688|
|0.3|61.885|62.022|
|0.4|59.969|60.339|
|0.5|59.557|60.196|
|0.6|57.603|58.059|
|0.7|57.524|58.073|
|8-rate mean|60.432|60.879|
|High missing (.5/.6/.7)|58.228|58.776|

|Fold|8-rate W-F1 (%)|High-missing W-F1 (%)|
|---|---:|---:|
|1|60.082|57.047|
|2|63.168|61.616|
|3|57.293|55.375|
|4|58.395|56.362|
|5|63.221|60.740|

No random-seed variance/significance claim; folds are held-out sessions,
not independent random-seed replications. No mechanism claim or improvement
claim over historical Flat, whose selection_metric is unrecorded.
INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.
