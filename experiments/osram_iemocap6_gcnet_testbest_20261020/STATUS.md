# Status:30/30 complete; final paired report verified

## Progress inspection2026-10-20

30/30 runs complete, zero running/pending/failed. All five folds and all three
seeds completed 100 epochs. The complete-only paired report passed initialization,
training-mask, evaluation-cohort and exact selected-epoch prediction checks.
Independent verification recomputed ACC/UA/W-F1/CE from all 240 selected prediction
files; maximum absolute metric error was 2.842170943040401e-14. Every selected epoch
matches the earliest per-rate TEST ACC maximum in its complete training history.
Final reports: RESULT.md, METRICS.csv, COMPACT_SUMMARY.json, FINAL_VERIFICATION.json.
Fivefold/three-seed eight-rate ACC: Flat60.7083, Router60.9170; UA:58.7063,59.2870.
Historical launch details below remain snapshots,not the current queuecounts.

Approved finite rerun:30fresh IEMOCAP-6 runs,100epochs,GCNet missing,
per-rate TEST ACC BEST, same-call UA/W-F1. Original full training folds restored.
Existing MOSI reruns remain independent and untouched.53targeted tests and
compileall passed. RealGPU Joint smoke completed:finite/nonzero gradients,
backbone/Router updates, full model/Adam/RNG restore and exact selected prediction
reload passed. Peak allocated5287.53125MiB. Smoke is not a performance result.

Cleanup complete:62model files permanently removed,10,218,454,510bytes.
483preserved evidence files rehashed unchanged;135original baseline checkpoints
retain identical size/mtime. See CLEANUP_PLAN.json and CLEANUP_RECEIPT.json.
Remote receipt SHA256:977bc91c3bb7a10e9d0cbbd07df263a8fd642ebf969d321024ab807181189656.

Source commit5bab8e76c2f85de8406822e5cd5836988690a2cb, frozen source archive
SHA256594e487fef66a2ea55149f1bb2d3cc686210d415f4d0b84835eb7017b552954f.
Remote root:`/data2/yb/remote_experiments/osram_iemocap6_gcnet_testbest_20261020`.
Source:`source_5bab8e7`; output:`execution_v1`; tmux:`iemo6_testbest_5bab8e7`.
Physical GPU0/2/3,UUID-bound, maximum2children per queue/card subject to
resources. No completed training score claimed.

Controller PID755360; six initial fold1 workers:
seed66 Flat756031/Router756098, seed67 Flat756202/Router756388,
seed68 Flat756486/Router756575.30jobs planned,6running processes,24pending;
the first process inspection caught initialization, not completed epochs.
Subsequent inspection confirmed all six effective configs explicitly use
test-oracle-per-rate/accuracy/gcnet-random-mask-v1 and all123original fold1
training conversations, adapter_holdout=False. No worker error recorded.
See QUEUE_AT_LAUNCH.json and SMOKE.json. The queue automatically fills slots
and writes the final result only after all30runs and paired checks complete.
