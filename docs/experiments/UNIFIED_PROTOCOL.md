# GCNet unified Baseline/JEPA protocol

This is the only protocol for new results. Historical sweeps produced before
this file are diagnostic evidence, not formal Baseline/JEPA comparisons.

## Fixed environment and data protocol

- Interpreter: `/data2/yb/reproduction_envs/gcnet-official/bin/python`
- Environment lock: `environments/gcnet-official/environment.yml`
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- IEMOCAP: five leave-one-session-out test folds; validation conversations are
  selected only from the four non-test sessions with fraction `0.1`.
- MOSI/MOSEI: the official train/validation/test video memberships are retained.
- Training masks vary deterministically by conversation and epoch. Validation
  and test masks are fixed at epoch zero.
- Validation Weighted-F1 selects the checkpoint. Test is called exactly once
  after the best checkpoint is restored.

The trainer writes immutable evidence under
`OUTPUT/run_records/RUN_ID/`: `fold_metrics.json` and one manifest per executed
fold. Compare a Baseline/JEPA pair with:

```bash
PY=/data2/yb/reproduction_envs/gcnet-official/bin/python
$PY scripts/audit_paired_runs.py BASELINE_MANIFEST.json JEPA_MANIFEST.json
```

An audit failure invalidates the pair even when its scores look plausible.

## Canonical command shape

Set the data root and keep outputs outside the source tree when possible:

```bash
export GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_mosi_collapse_diag_20260820/dataset
export PYTHONPATH=/data2/yb/paper/GCNet_mosi_collapse_diag_20260820
PY=/data2/yb/reproduction_envs/gcnet-official/bin/python
COMMON="--audio-feature wav2vec-large-c-UTT --text-feature deberta-large-4-UTT --video-feature manet_UTT --base-model LSTM --windowp 2 --windowf 2 --hidden 200 --lr 0.001 --dropout 0.5 --batch-size 32 --stability-aux-mask-rate 0.1 --stability-recon-weight 0.01"
```

Single IEMOCAP fold, Baseline:

```bash
CUDA_VISIBLE_DEVICES=5 $PY -u -m gcnet_modality_jepa.train_gcnet \
  $COMMON --dataset IEMOCAPSix --fold 5 --epochs 100 --seed 66 \
  --mask-type constant-0.3 --loss-recon --jepa-weight 0 \
  --model-variant addon --output-dir /data2/yb/experiments/baseline
```

The paired JEPA command changes only method-specific flags and output:

```bash
CUDA_VISIBLE_DEVICES=5 $PY -u -m gcnet_modality_jepa.train_gcnet \
  $COMMON --dataset IEMOCAPSix --fold 5 --epochs 100 --seed 66 \
  --mask-type constant-0.3 --jepa-weight 0.1 \
  --model-variant replacement --output-dir /data2/yb/experiments/jepa
```

Omit `--fold` for all five IEMOCAP folds. MOSI and MOSEI use one official
split and therefore do not take an IEMOCAP fold selector.

## Determinism boundary

Python, NumPy, Torch, CUDA, sample order, masks, shared initialization, and
training stochasticity are explicitly seeded. CUDA `torch_scatter`/PyG graph
aggregation uses scatter/atomic kernels for which this stack has no strict
deterministic implementation. `--strict-deterministic` is therefore a
diagnostic mode that intentionally errors at that operation, not the formal
training command. Formal claims use paired seeds, paired deltas, mean ± standard
deviation, and collapse counts.

GPU 4 on `biggpu` is excluded from all orchestration.

## Smoke gate before formal runs

Every code or protocol change must first pass two epochs for Baseline and JEPA
on IEMOCAP-Four, IEMOCAP-Six, CMU-MOSI, and CMU-MOSEI. Required evidence:

- finite train/validation/test losses;
- `test_call_count == 1`;
- valid and disjoint splits;
- matching paired feature, split, order, mask, shared-init, and stability fields;
- successful `audit_paired_runs.py` for each dataset pair.

Only after all eight smoke jobs pass may the 10-seed formal queue start.
