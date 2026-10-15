# B6 structural HPO, fixed trial16 training configuration

INTERNAL TEST-ORACLE SCREEN; seed66 only

| Trial | Rank | Steps | State | Test-best ACC | UA | Paired ACC | UA | Objective |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | 14 | COMPLETE | 60.7328 | 35.0750 | 60.6226 | 35.2006 | 0.0461 |
| 1 | 1 | 8 | COMPLETE | 60.9243 | 34.4507 | 60.6322 | 34.5892 | -0.1927 |
| 2 | 1 | 24 | COMPLETE | 60.9339 | 37.4209 | 60.7998 | 37.1884 | 0.9476 |
| 3 | 2 | 14 | COMPLETE | 61.2165 | 35.9354 | 60.9004 | 35.8789 | 0.4841 |
| 4 | 3 | 14 | COMPLETE | 60.8860 | 36.2510 | 60.6753 | 35.8786 | 0.3489 |
| 5 | 1 | 26 | COMPLETE | 60.8238 | 36.5129 | 60.7328 | 36.4928 | 0.6291 |
| 6 | 2 | 28 | COMPLETE | 60.7280 | 35.9400 | 60.4598 | 35.3781 | 0.0194 |
| 7 | 3 | 22 | COMPLETE | 60.9243 | 35.4790 | 60.4598 | 34.9599 | -0.1479 |
| 8 | 1 | 4 | COMPLETE | 60.9962 | 34.3689 | 60.7759 | 34.3893 | -0.1865 |
| 9 | 1 | 20 | COMPLETE | 60.5603 | 35.7554 | 60.5268 | 35.8175 | 0.2354 |
| 10 | 2 | 20 | COMPLETE | 60.6705 | 35.4924 | 60.4119 | 35.5429 | 0.0566 |
| 11 | 3 | 8 | COMPLETE | 61.2165 | 34.5950 | 60.9914 | 34.6774 | 0.0581 |
| 12 | 1 | 22 | COMPLETE | 60.6274 | 37.0699 | 60.7136 | 36.9243 | 0.7902 |
| 13 | 1 | 10 | COMPLETE | 60.7375 | 35.2603 | 60.6849 | 35.1258 | 0.0536 |
| 14 | 2 | 4 | COMPLETE | 60.4262 | 36.9664 | 60.6082 | 36.7316 | 0.6499 |
| 15 | 3 | 18 | COMPLETE | 60.5651 | 35.4209 | 60.3592 | 35.4252 | -0.0221 |
| 16 | 1 | 28 | COMPLETE | 60.6178 | 35.7404 | 60.3065 | 35.3282 | -0.0925 |
| 17 | 1 | 16 | COMPLETE | 60.9722 | 35.0017 | 60.5460 | 34.6980 | -0.2009 |
| 18 | 2 | 24 | COMPLETE | 60.6370 | 35.5299 | 60.5172 | 35.4768 | 0.0934 |
| 19 | 3 | 10 | COMPLETE | 61.1159 | 36.1545 | 61.1207 | 36.1476 | 0.7238 |

Best paired objective trial: 2
Highest Test-best ACC trial: 3

All scores are test-selected exploration, not independent test performance.
Trial0 repeats rank1/steps14 at this immutable source. No old score is counted as a new run.

## Verified completion

20/20unique configurations COMPLETE,0failures,all50epochs. Finished2026-10-15
07:12:29UTC; controller wall time15266.08seconds (4h14m26s), including smoke,
training, queue waits and replay. This is shared-server wall time, not isolated
model throughput or exclusive GPUhours. Source remains
`9048ffdcbeb30006c49e45c0d1387cada4d85242`.

All160BEST weights remain on biggpu. All160exact selected-epoch artifact hashes
were checked, and ACC/7class UA/weighted F1 were independently recomputed from
their labels/logits. All20history files contain50epoch records. No new inference
or training was needed for this final audit. All runs retain fixed trial16
learning rate/weight decay/dropout and original GCNet missing protocol.
CSV400rows preserve per-rate,mean,high scores for Test-best and paired-on;
JSON additionally retains paired-off,corrections/harms,cost and provenance.

## Main configurations (eight-rate means, %)

| Selection criterion | Trial | Rank | Steps | Test-best ACC | Matching UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Same-source original B6 structure control | 0 | 1 | 14 | 60.733 | 35.075 | 56.674 |
| Highest Test-best ACC (tie; higher UA) | 3 | 2 | 14 | 61.216 | 35.935 | 57.254 |
| Highest Test-best ACC (tie; lower UA) | 11 | 3 | 8 | 61.216 | 34.595 | 56.755 |
| Highest UA,W-F1,and registered paired objective | 2 | 1 | 24 | 60.934 | 37.421 | 57.570 |
| Highest common-mask paired ACC,not highest Test-best ACC | 19 | 3 | 10 | 61.116 | 36.155 | 57.150 |

Trial2is still the registered objective winner (0.947573pp). Trial3and11tie
exactly in Test-best ACC; reporting trial3alone without acknowledging the tie
would omit a completed result. Trial19's highest **paired** ACC is61.120690,
with paired UA36.147557/W-F1 57.114212; do not insert those scores in Test-best
columns. Trial2paired scores:60.799808/37.188429/57.375037. Trial3paired scores:
60.900383/35.878905/56.965013.

## High missing (0.5/0.6/0.7), Test-best

| Trial | ACC | UA | W-F1 |
|---:|---:|---:|---:|
| 0 | 56.833 | 29.251 | 51.023 |
| 2 | 56.794 | 30.107 | 51.371 |
| 3 | 57.216 | 29.017 | 50.949 |
| 11 | 57.178 | 28.607 | 50.691 |
| 19 | 57.190 | 31.036 | 51.810 |

No single configuration dominates all metrics/conditions. The historical accepted
Flat3seed mean remains60.693/36.255/57.045; it is a reference, not a matched
multi-seed control for this seed66search. This study did not add new Flat or
structural seed67/68runs.

## Direct branch contribution remains negligible

Across all20runs, maximum absolute same-checkpoint B6-on/off change is
ACC0.014368pp,UA0.038970pp,W-F1 0.019216pp. Trial2has1correction/0harms across
20880rate-observations (8x2610not distinct utterances); trial3has0/0. Trial11has
0/1 and trial19has1/0. This intervention changes only the second block, not CHM,
original Local Skip, Flat/head weights or masks. Tiny metric changes do not
imply identical continuous logits.

Therefore the selected model scores are valid exploratory observations, but
cannot establish a substantial direct decision contribution of this B6branch.
Training-trajectory effects and numerical/retraining variation remain possible
and are not separated here. Larger rank/depth is not monotonically better.

## Resource and follow-up boundary

Used verified UUIDs for hostGPU0/2/3,neverGPU4. Existing jobs were not signalled.
Worker-reported peak allocated memory ranges631.905–945.868MiB; the separately
recorded worst-length batch smoke peaked1470.322MiB. Learned B6parameter counts
range9–57(`2*steps+1`); rank changes factor compute,not learned vector length.
No performance or significance claim follows from the smoke.

The authorized20trial study ended automatically. Do not extend the budget or
start structural multi-seed runs without a new request. Rank1/steps24is the
most balanced candidate for a potential later check; rank2/steps14prioritizes
ACC. Given negligible direct on/off benefit, neither is established as a
successful second contribution. Retain negative results and all exact evidence.
