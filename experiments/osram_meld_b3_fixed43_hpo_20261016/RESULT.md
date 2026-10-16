# MELD B3: fixed four tokens / three exchange rounds

INTERNAL TEST-ORACLE SCREEN; seed66 only

| Trial | bottlenecks | rounds | dim | heads | ffn_hidden | lr_multiplier | State | Test-best ACC | UA | Paired ACC | UA | Objective |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 4 | 3 | 256 | 4 | 256 | 1 | COMPLETE | 60.6657 | 36.0196 | 60.3975 | 36.0264 | 0.2414 |
| 3 | 4 | 3 | 128 | 8 | 128 | 1 | COMPLETE | 60.6992 | 34.8201 | 60.5029 | 34.4483 | -0.3266 |
| 4 | 4 | 3 | 128 | 4 | 128 | 0.3 | COMPLETE | 60.7854 | 35.3566 | 60.9722 | 35.4710 | 0.3641 |
| 5 | 4 | 3 | 128 | 4 | 256 | 0.3 | COMPLETE | 60.5843 | 34.8412 | 60.4167 | 34.6833 | -0.2843 |

Best paired objective trial: 4
Highest Test-best ACC trial: 4

All scores are test-selected exploration, not independent test performance.
Only completed 4/3 observations train TPE; all 28 prior tuples excluded. Exactly four new jobs.
