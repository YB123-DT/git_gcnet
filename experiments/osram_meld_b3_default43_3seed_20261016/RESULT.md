# MELD default B3 4/3: three seeds

INTERNAL TEST-ORACLE THREE-SEED REPLICATION; seed66 selected configurations

| Configuration | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| B3_default43 | 66 | 60.757 | 37.358 | 57.396 |
| B3_default43 | 67 | 60.551 | 34.857 | 56.419 |
| B3_default43 | 68 | 60.354 | 37.244 | 57.219 |
| B3_default43 | mean ± sample SD | 60.554 ± 0.201 | 36.486 ± 1.412 | 57.011 ± 0.520 |

Per-rate and high-missing results and full source/configuration provenance are in SUMMARY.csv/JSON.
No new seed66 training, architecture change, validation selection, or additional hyperparameter search.
Seed66 was used for configuration search; this three-seed mean is not independent test evidence.
Original seed66 and new workers use immutable737fdb1. Replication changes only an explicit seed admission guard and the launcher; numerical model/trainer/sampler source hashes must match.
