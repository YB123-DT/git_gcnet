# Causal ComP CMU-MOSI Seed-66 Run

## Scope

This experiment runs the utterance-level temporal-causal ComP variant on CMU-MOSI at missing rates `0.0` through `0.7`. It preserves the official feature set, optimizer, batch size, two-stage 150/150 epoch schedule, and test-peak reporting protocol.

The source baseline is upstream ComP commit `28192d3a5683543d7383e40898f9a98d1f114a08`, vendored at `external_repos/ComP_causal`.

## Environment and data

- Host: local experiment server (`user23`)
- Conda environment: `/home/yangbin/miniconda3/envs/comp-repro`
- Python: 3.8.20
- PyTorch: 1.12.0+cu113
- Data root: `/data2/yb/paper/GCNet_TPAMI/dataset/CMUMOSI`
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- Formal output: `/data2/yb/paper/05_reproduction/runs/ComP_causal/seed66/CMUMOSI`

## Verification

Run the deterministic causal checks with:

```bash
/home/yangbin/miniconda3/bin/conda run --no-capture-output -n comp-repro \
  python external_repos/ComP_causal/tests/test_temporal_causality.py
```

The checks cover causal attention, causal prefix prototypes, both prompt stages, end-to-end future perturbation with missing modalities, final-prefix equivalence, and prototype-gradient updates.

The real-data smoke command is:

```bash
experiments/comp_causal_seed66/run_smoke.sh 3
```

On 2026-10-10 it completed one Stage-1 and one Stage-2 epoch on physical GPU 3 without NaN, OOM, or traceback. The smoke output is kept under `/data2/yb/paper/05_reproduction/runs/ComP_causal/smoke/CMUMOSI`; it is not performance evidence.

## Formal workers

Each worker receives a physical GPU index followed by one or more missing rates:

```bash
experiments/comp_causal_seed66/run_seed66_worker.sh 3 0.0 0.4
```

Every rate has an isolated directory, `run.log`, `run_manifest.txt`, official result text, and runtime-generated log tree. The worker skips a setting only when its result contains `Folder avg:`.
