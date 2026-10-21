# V3 finite overnight queue

INTERNAL TEST-ORACLE SCREEN

Implementation and20 CPU driver tests complete. Remote GPU smoke and actual
launch status are recorded separately in LAUNCH.json after execution.

Remote20 tests and all three real GPU smokes now passed. The persistent queue
has started formal training; startup evidence records three running jobs and
eight pending jobs. Pending runs automatically fill available healthy GPU0/2
slots. This is a timestamped startup record, not completion of the11 runs.

Actual immutable numerical source: `1e44ef3ce36feab63de1e57b7175a1589688c7a2`.
biggpu output:
`/data2/yb/remote_experiments/osram_v3_classification_20261021/execution_v1`.
tmux socket/session: `osram_v3_classification` / `seed66_1e44ef3`.

| Dataset | seed | folds | epochs | LR | scope |
|---|---:|---|---:|---:|---|
| MELD |66|official|50|1e-5|one new V3 run|
| IEMOCAP-4 |66|1–5|100|1e-3|five new V3 runs|
| IEMOCAP-6 |66|1–5|100|1e-3|five new V3 runs|
| MOSEI |66|official|60|1e-3|existing V3 continues; no duplicate|

No performance results yet for new classification jobs. Current16/old method
results are not substituted. GPU0 max4/GPU2 max8 are resource-dependent caps,
not claims that all slots have already launched. All BEST and full last states
are retained; no old checkpoint is deleted or overwritten.
