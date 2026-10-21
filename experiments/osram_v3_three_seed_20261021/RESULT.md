# V3 three-seed completion status

INTERNAL TEST-ORACLE SCREEN

Requested: reuse12 existing seed66 runs, add24 missing67/68 runs. No final
three-seed performance is available for the new suite yet. Actual remote queue
launch/wait state is retained in LAUNCH.json after execution.

The remote supplement controller has actually started under tmux:
`osram_v3_three_seed` / `supplement_d6c802e`, PID326112 at launch.
Its24 jobs are pending behind the still-running seed66 controller. They will
start automatically after the predecessor drains and resources permit; this
is not a claim that all24 training workers are already running.

Source: `d6c802ecfe105ffce1ef0badfdbcf697b312f71d`.
Output: `/data2/yb/remote_experiments/osram_v3_three_seed_20261021/execution_v1`.

| Dataset | retained seed66 | newly added seeds67/68 |
|---|---:|---:|
| MOSEI |1|2|
| MELD |1|2|
| IEMOCAP-4 |5|10|
| IEMOCAP-6 |5|10|

Original data/missing/optimizer/task budgets remain unchanged. Existing seed66
jobs are not restarted; the new persistent controller automatically takes over
after their controller drains, with GPU0/2 total caps4/8. No currently pending
run is described as training or completed.
