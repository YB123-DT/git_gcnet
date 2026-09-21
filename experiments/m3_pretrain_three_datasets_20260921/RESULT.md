# Three-dataset M3 pretraining result

This report records the completed five-seed utterance-level pretraining run.
The model shares one set of modality projectors, EMA projectors, and one
SourceOnlyM3Predictor across CMU-MOSI, CMU-MOSEI, and IEMOCAP.  It does not run
OSRAM, GCNet, emotion classification, or test data.

## Protocol

- Seeds: `66, 67, 68, 69, 70`
- Epochs: `100`
- Dataset sampling: uniform dataset-level schedule
- Source patterns: `A`, `T`, `V`, `AT`, `AV`, `TV`
- IEMOCAP split: fold/session 5 as configured by the runner
- Reported values below are the final validation metrics after epoch 100.
- Raw per-seed train/validation metrics are in `seed_*/metrics.json`.

## Validation centered cosine (mean ± sample SD)

| Dataset | A→T | V→T | AV→T | T→A | T→V |
|---|---:|---:|---:|---:|---:|
| CMU-MOSI | 0.1799 ± 0.0322 | 0.0474 ± 0.0369 | 0.1489 ± 0.0278 | 0.1774 ± 0.0149 | 0.0363 ± 0.0250 |
| CMU-MOSEI | 0.2855 ± 0.0097 | 0.1980 ± 0.0148 | 0.2958 ± 0.0187 | 0.2729 ± 0.0121 | 0.1799 ± 0.0122 |
| IEMOCAP (6) | 0.3298 ± 0.0308 | 0.1823 ± 0.0191 | 0.2972 ± 0.0147 | 0.3449 ± 0.0085 | 0.1564 ± 0.0089 |

## MOSI per-seed validation routes

| Seed | A→T | V→T | AV→T | T→A | T→V |
|---:|---:|---:|---:|---:|---:|
| 66 | 0.2160 | 0.0157 | 0.1790 | 0.1809 | 0.0600 |
| 67 | 0.2084 | 0.0449 | 0.1579 | 0.1689 | 0.0606 |
| 68 | 0.1705 | 0.1008 | 0.1181 | 0.1659 | 0.0353 |
| 69 | 0.1376 | 0.0642 | 0.1210 | 0.2022 | 0.0237 |
| 70 | 0.1670 | 0.0116 | 0.1683 | 0.1693 | 0.0019 |

## Interpretation

The five-seed result confirms that the joint predictor does not currently
generalize well to MOSI.  Its validation centered cosine is weak for every
direction and especially weak for `V→T` and `T→V`, while the corresponding
training values are much higher.  For seed 66, for example, `T→V` falls from
`0.6624` on train to `0.0600` on validation; the other MOSI routes show the
same train/validation separation rather than an isolated failure of one route.

The other datasets do show usable, but still imperfect, validation transfer
when averaging all nine routes:

| Dataset | Train mean | Validation mean | Validation − train |
|---|---:|---:|---:|
| CMU-MOSI | 0.6900 | 0.1433 | -0.5467 |
| CMU-MOSEI | 0.4253 | 0.2746 | -0.1507 |
| IEMOCAP (6) | 0.5567 | 0.2941 | -0.2626 |

Thus MOSEI has the smallest train/validation gap and the most stable transfer.
IEMOCAP has a larger absolute gap, but its validation alignment remains
consistently non-zero across seeds.  These are signs of partial
generalization, not strong sample-level prediction.  In particular, the
validation `V→T`/`T→V` routes remain weak on both datasets (MOSEI `0.1980` /
`0.1799`; IEMOCAP `0.1823` / `0.1564`).

MOSEI and IEMOCAP show more stable, non-zero validation alignment, but this
does not rescue the MOSI setting.  The likely confound is the current
dataset-uniform, fixed-step schedule: MOSI has only two training batches while
the joint run performs 71 sampled steps per epoch, so its small training set is
reused heavily.  This is a schedule/data-budget diagnosis, not evidence that
cross-dataset JEPA is impossible.

This run is therefore **not a usable generalized MOSI predictor**.  No OSRAM
or classification conclusions should be drawn from it.

## Artifacts

- `seed_66/metrics.json` through `seed_70/metrics.json`: lightweight raw metrics
- `run.py`, `evaluate.py`, `README.md`: reproducible runner/evaluator
- Checkpoints are intentionally excluded from Git; they remain local under the
  corresponding `seed_*` directories.
