# PLCI Single-View Stage-1 Runbook

## Scope

- Dataset: `IEMOCAPSix`
- Fold: `5`
- Rates: `0.0`, `0.5`, `0.7`
- Seeds: `66`, `67`, `68`, `69`, `70`
- New training jobs: `15`
- Dual-View jobs rerun: `0`
- Original jobs rerun: `0`

The primary A/B is Single-View versus the existing Dual-View PLCI runs. Both
use `gcnet_modality_jepa.train_gcnet`, `evaluation_protocol=official`,
`stability_recon_weight=0`, and the same deterministic conversation-mask
schedule. Original is retained as secondary context.

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

## Expansion Gate

Do not launch other missing rates or datasets until all 15 jobs finish and the
Single-vs-Dual paired table is audited. Expansion requires no collapse and no
material loss at both active rates `0.5` and `0.7`.
