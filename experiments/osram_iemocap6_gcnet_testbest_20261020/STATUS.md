# Status: GPU smoke passed; six workers launched,24jobs pending

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
