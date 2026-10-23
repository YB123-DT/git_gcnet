# Live handoff — training is not a result

2026-10-23 05:30 UTC. INTERNAL TEST-ORACLE SCREEN.

Both GPU smokes passed on biggpu physical GPU7; peak allocated3,194.83MiB,
four optimizer updates each. Original initialization and exact test masks match
the accepted Flat reference. Full model/Adam/RNG/prediction reload checks passed.
The TRAIN bank contains1,284 utterances from52 conversations; both arms have the
same bank hashes,13,782,753 total parameters,272,960 added parameters.

Both formal MOSI seed66 jobs are running concurrently and have completed epoch1
of150. No formal final scores or improvement claims are available yet.

- feature-only PID1399503
- label-aware PID1400641
- Controller PID1386512; tmux `exemplar_mosi_20261023`
- Immutable training source `d6f6eb19d9ef68403dcfe23a18a7ef66712a9602`
- Output `/data2/yb/remote_experiments/osram_exemplar_mosi_20261023/execution_v1`
- Queue `QUEUE.json`; worker logs `train_MOSI66_{feature-only,label-aware}.log`

The controller refreshes RESULT.md/REPORT.json/METRICS.csv automatically. Only
completed150-epoch runs enter the score table. Per-rate BESTs, same-call masks
and predictions and full last checkpoints remain remote. No old jobs were
stopped, no protected data/checkpoints were changed, no HPO/seeds were added.

Accepted Flat66 reference: eight-rate W-F1 79.2490865%, high-missing W-F1
73.4097921%. This is the current GCNet-missing reference, not old80.7 results.

Independent specification review and code-quality review passed. The latter
found one recovery-admission flag race; it was fixed and regression-tested
before the immutable source was deployed. All32 implementation/regression
checks passed (11new model,7runner,14original OSRAM).
