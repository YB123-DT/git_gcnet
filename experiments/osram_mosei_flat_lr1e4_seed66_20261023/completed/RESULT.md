# MOSEI original Flat constant1e-4 three-seed completion

INTERNAL TEST-ORACLE SCREEN. Only seed66 is new;67/68 reused read-only.
60epochs,GCNet-random-mask-v1,original Flat;each rate keeps its original TEST-best W-F1 checkpoint.
Masks/seed-specific initialization follow the existing low-LR protocol. No new inference or reselection.

| Seed | Eight-rate ACC | W-F1 | High ACC | High W-F1 |
|---:|---:|---:|---:|---:|
| 66 | 83.5099 | 83.3650 | 80.4531 | 80.1083 |
| 67 | 83.4274 | 83.2445 | 80.2880 | 80.0437 |
| 68 | 83.5718 | 83.3268 | 80.3797 | 79.9281 |

Three-seed sample mean/SD and source identities are in RESULT.json; selected metrics/epochs in PER_RATE.csv.
Eight-rate mean can combine eight checkpoints and is not single-model deployment performance.
Exploratory test selection, not independent generalization evidence. No automatic new models/HPO/seeds.
