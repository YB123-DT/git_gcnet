# MOSI-only M3 pretraining control

This is the matched single-dataset control for the three-dataset utterance-
level M3 pretraining run.  It uses the same `ThreeDatasetJEPA` implementation,
projector/predictor dimensions, optimizer, masks, seed list, and 100-epoch
budget.  The only data change is that training and validation contain CMU-MOSI
only.

## Update-budget matching

The joint run uses 71 optimizer steps per epoch and samples datasets uniformly,
so MOSI receives approximately one third of those steps.  This control uses
24 steps per epoch, matching the MOSI optimizer-update exposure rather than
silently giving MOSI three times more updates.  All six source patterns and all
nine directed routes are unchanged.

- Seeds: `66, 67, 68, 69, 70`
- Train/validation only; no test data
- Final metrics after epoch 100

## Five-seed validation comparison

Values are centered cosine, mean ± sample SD over five seeds.  The delta is
`MOSI-only − three-dataset joint` on the same MOSI validation protocol.

| Route | MOSI-only | Joint | Delta |
|---|---:|---:|---:|
| A→T | 0.1361 ± 0.0232 | 0.1799 ± 0.0322 | −0.0437 |
| A→V | 0.1032 ± 0.0577 | 0.1412 ± 0.0234 | −0.0380 |
| T→A | 0.1336 ± 0.0318 | 0.1774 ± 0.0149 | −0.0438 |
| T→V | 0.0202 ± 0.0106 | 0.0363 ± 0.0250 | −0.0161 |
| V→A | 0.1015 ± 0.0191 | 0.1652 ± 0.0373 | −0.0637 |
| V→T | 0.0434 ± 0.0230 | 0.0474 ± 0.0369 | −0.0041 |
| AT→V | 0.0855 ± 0.0222 | 0.1393 ± 0.0288 | −0.0538 |
| AV→T | 0.1146 ± 0.0125 | 0.1489 ± 0.0278 | −0.0343 |
| TV→A | 0.1792 ± 0.0096 | 0.2542 ± 0.0432 | −0.0750 |

## All-route aggregate

| Seed | MOSI-only validation | Joint validation | Joint − MOSI-only |
|---:|---:|---:|---:|
| 66 | 0.1092 | 0.1510 | +0.0418 |
| 67 | 0.0977 | 0.1433 | +0.0456 |
| 68 | 0.0969 | 0.1389 | +0.0420 |
| 69 | 0.1137 | 0.1479 | +0.0341 |
| 70 | 0.0920 | 0.1354 | +0.0434 |
| Mean | 0.1019 | 0.1433 | +0.0414 |

## Train/validation behavior

The MOSI-only model memorizes its small training set more strongly:

| Control | Train all-route mean | Validation all-route mean | Validation − train |
|---|---:|---:|---:|
| MOSI-only | 0.8423 | 0.1019 | −0.7404 |
| Joint | 0.6900 | 0.1433 | −0.5467 |

Therefore the three-dataset run is not failing because it has no benefit for
MOSI.  Under matched MOSI update exposure, joint pretraining improves all nine
validation directions and raises the all-route mean by `0.0414` centered-cosine
points.  The gain is still modest and does not make the predictor strong: the
MOSI validation alignment remains low, especially `T→V` and `V→T`.

The correct conclusion is that additional datasets act as regularization and
provide a broader cross-modal training signal, while the MOSI-only predictor
overfits.  This is a representation-quality comparison, not an emotion
classification result.

Raw per-seed metrics are in `seed_*/metrics.json`; checkpoints are intentionally
kept local and excluded from Git.
