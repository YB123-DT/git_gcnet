# GCNet Official-Protocol Four-Dataset Rerun — 2026-08-20

## Frozen matrix

- Datasets: IEMOCAPFour, IEMOCAPSix, CMUMOSI, CMUMOSEI.
- Methods: GCNet baseline and GCNet+JEPA.
- Missing rates: 0.0–0.7.
- Seeds: 66–75.
- IEMOCAP: held-out Session 5 only (`fold=5`).
- Total: 640 runs.
- Environment: `/data2/yb/reproduction_envs/gcnet-official/bin/python`.
- GPUs: 0, 1, 2, 3, 5; GPU4 excluded; at most three concurrent jobs per GPU.

## Evaluation behavior

Formal runs use `--evaluation-protocol official`. Every epoch evaluates both
validation and test. The selected epoch is determined only by validation
Weighted-F1, and the reported test payload is the payload already computed at
that epoch. For IEMOCAP, the held-out Session 5 is used for both validation and
test, matching the original GCNet evaluation topology. The explicit `strict`
mode remains available for internal validation plus one final test, but its
scores are not mixed with this rerun.

Evaluation masks vary deterministically by epoch and split in official mode.
Baseline and JEPA share data order, masks, common stability configuration,
training seed, and shared initialization evidence.

## Verification before launch

- Fixed official environment: 151 unit/integration tests passed.
- Real-data smoke: 8/8 jobs completed with return code 0.
- Paired manifest audit: 4/4 dataset pairs passed.
- Every smoke manifest recorded `evaluation_protocol=official`,
  `test_call_count=2`, and `epochs_completed=2`.
- IEMOCAP smoke manifests recorded `fold=5`.
- Shared initialization hashes matched within all four baseline/JEPA pairs.

Two-epoch smoke Weighted-F1 values are infrastructure evidence only:

| Dataset | Baseline | JEPA |
|---|---:|---:|
| IEMOCAPFour | 0.146240 | 0.187076 |
| IEMOCAPSix | 0.119659 | 0.155451 |
| CMUMOSI | 0.537075 | 0.561585 |
| CMUMOSEI | 0.743367 | 0.724339 |

These are not scientific results because the smoke uses only two epochs.

## Formal launch

Output root:

`/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820`

Task manifest checks:

- 640 total and 640 unique identities;
- 160 tasks per dataset;
- 320 tasks per method;
- 320 IEMOCAP commands with `--fold 5`;
- zero tasks assigned to GPU4.

Launch command:

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -u \
  scripts/run_official_missing_sweep.py \
  --output-root /data2/yb/experiments/gcnet_official_4dataset_10seed_20260820 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python \
  --gpus 0,1,2,3,5 --jobs-per-gpu 3 --epochs 100
```

Scheduler PID at launch: `3648581`.

The runner is resumable. A run is skipped only when `status.json` has return
code zero and a fold manifest exists. Any training failure stops new jobs; all
completed outputs remain reusable.
