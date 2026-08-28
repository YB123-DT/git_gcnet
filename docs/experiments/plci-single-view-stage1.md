# PLCI Single-View Stage-1 Runbook

## Scope

- Dataset: `IEMOCAPSix`
- Fold: `5`
- Initial rates: `0.0`, `0.5`, `0.7`
- Continuation rates: `0.1`, `0.2`, `0.3`, `0.4`, `0.6`
- Seeds: `66`, `67`, `68`, `69`, `70`
- Initial training jobs: `15`
- Continuation training jobs: `25`
- Total Single-View jobs after continuation: `40`
- Dual-View jobs rerun: `0`
- Original jobs rerun: `0`

The primary output is the standalone Single-View performance curve over all
eight missing rates. Dual-View is not the acceptance baseline; its existing
results are retained only as an optional architecture ablation. Neither
Dual-View nor Original is rerun by this continuation.

## Remote Paths

- Code: `/data2/yb/paper/GCNet_TPAMI_single_view_dev`
- Python: `/data2/yb/reproduction_envs/gcnet-official/bin/python`
- Output: `/data2/yb/paper/experiments/plci_single_view_iemocap6_stage1_20260827`
- Dual-View control: `/data2/yb/paper/experiments/plci_jepa_iemocap6_20260826/formal`
- Original context: `/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/IEMOCAPSix`

## Scheduling

Use healthy GPUs `0,1,2`, with three lanes per GPU. GPU 4 remains excluded. Each
lane runs its assigned jobs sequentially, so a GPU has at most three concurrent
training processes.

## Launch

```bash
cd /data2/yb/paper/GCNet_TPAMI_single_view_dev
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  scripts/run_plci_single_view_iemocap6.py \
  --output-root /data2/yb/paper/experiments/plci_single_view_iemocap6_stage1_20260827 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python \
  --gpus 0 1 2 \
  --jobs-per-gpu 3 \
  --epochs 100
```

The runner refuses non-empty partial output directories. It writes
`command.json`, `train.log`, and `status.json` per job and treats a run as
complete only when status, `fold_metrics.json`, and exactly one top-level NPZ
archive agree.

## Stop

Only processes whose command contains both
`run_plci_single_view_iemocap6.py` or `plci_single_view_iemocap6_stage1_20260827`
belong to this experiment. Do not terminate other users or unrelated GCNet
runs.

## Initial Expansion Decision

The initial three-rate run completed without collapse and produced usable
standalone scores. Following the author decision, run the remaining five rates
before making a final method-level decision. Do not use a comparison with
Dual-View alone to close the Single-View route.

## Completed Results — 2026-08-27

All 15 Single-View jobs completed successfully. All 15 pairs matched the
existing Dual-View control in both conversation-mask schedule hash and shared
initialization hash. Values below are Fold-5 test weighted F1 in percent.

| Missing rate | Seed | Single-View | Dual-View | Paired delta |
| --- | ---: | ---: | ---: | ---: |
| 0.0 | 66 | 62.75 | 63.22 | -0.47 |
| 0.0 | 67 | 62.72 | 60.80 | +1.93 |
| 0.0 | 68 | 62.43 | 63.65 | -1.22 |
| 0.0 | 69 | 63.16 | 62.63 | +0.53 |
| 0.0 | 70 | 65.79 | 65.27 | +0.52 |
| 0.5 | 66 | 60.54 | 61.63 | -1.09 |
| 0.5 | 67 | 66.22 | 63.60 | +2.62 |
| 0.5 | 68 | 63.68 | 62.67 | +1.01 |
| 0.5 | 69 | 64.71 | 61.79 | +2.92 |
| 0.5 | 70 | 63.29 | 66.19 | -2.91 |
| 0.7 | 66 | 61.97 | 62.00 | -0.03 |
| 0.7 | 67 | 64.53 | 63.72 | +0.81 |
| 0.7 | 68 | 59.99 | 61.87 | -1.87 |
| 0.7 | 69 | 61.73 | 63.62 | -1.89 |
| 0.7 | 70 | 60.07 | 62.43 | -2.37 |

Five-seed summaries report mean plus or minus sample standard deviation.

| Missing rate | Single-View | Dual-View | Mean paired delta | Positive seeds | Paired t-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 63.37 +/- 1.38 | 63.11 +/- 1.62 | +0.26 | 3/5 | 0.652 |
| 0.5 | 63.69 +/- 2.09 | 63.18 +/- 1.86 | +0.51 | 3/5 | 0.671 |
| 0.7 | 61.66 +/- 1.85 | 62.73 +/- 0.89 | -1.07 | 1/5 | 0.158 |

The table above is retained as an optional architectural comparison, not as the
acceptance criterion for Single-View. No paired difference is statistically
significant with five seeds. The standalone scores at `0.5` and `0.7` justify
recording the complete eight-rate curve before making a final method-level
decision.

The `jepa_loss=0`, zero pattern counts, and zero EMA steps stored in
`fold_metrics.json` describe the final test-only pass, where JEPA is disabled.
They do not mean that JEPA was absent during training: the training logs contain
non-zero `train_loss3` at rates `0.5` and `0.7` (for example, seed 66 ends at
`0.0545` and `0.1103`, respectively). At rate `0.0`, Single-View has no missing
Natural targets, so its JEPA term is correctly zero.

## Current Decision

**CONTINUE — complete the Single-View missing-rate sweep.** Do not treat
Dual-View as the primary comparator. After all 40 Single-View runs finish,
report the eight-rate five-seed curve on its own; any baseline comparison must
be labelled separately and audited for protocol compatibility.

## Full-Rate Continuation Launch — 2026-08-28

The same runner was extended with explicit `--missing-rates` and `--seeds`
selectors. A dry run generated 40 identities, inherited the 15 completed jobs,
and selected exactly 25 new jobs at rates `0.1`, `0.2`, `0.3`, `0.4`, and
`0.6`. It generated zero Original commands and does not require a Dual-View
control audit.

```bash
cd /data2/yb/paper/GCNet_TPAMI_single_view_dev
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  scripts/run_plci_single_view_iemocap6.py \
  --output-root /data2/yb/paper/experiments/plci_single_view_iemocap6_stage1_20260827 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python \
  --gpus 0 1 2 \
  --jobs-per-gpu 3 \
  --epochs 100 \
  --missing-rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --seeds 66 67 68 69 70
```

Launcher log:
`/data2/yb/paper/plci_single_view_all_rates_launcher_20260828.log`.

## Operational Record

The first launcher attempt exited before training because the isolated remote
code directory contained labels but not the large feature directories. Its logs
were preserved at
`/data2/yb/paper/experiments/plci_single_view_iemocap6_stage1_20260827_startup_failed_features_missing`.
The isolated directory now links its `dataset/IEMOCAP/features` path to the
shared feature store. A real Fold-5 batch was loaded before the successful
launcher was started.
