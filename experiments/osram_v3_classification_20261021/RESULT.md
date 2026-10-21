# V3 finite overnight queue

INTERNAL TEST-ORACLE SCREEN

Implementation and20 CPU driver tests complete. Remote GPU smoke and actual
launch status are recorded separately in LAUNCH.json after execution.

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
