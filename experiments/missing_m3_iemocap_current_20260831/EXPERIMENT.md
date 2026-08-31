# Current Missing-M3 on IEMOCAP-6/4

## Question

The previous IEMOCAP results used the early Missing-M3 implementation. This rerun
measures the current formal model after the observed-set and mixed-rate training
changes, without changing the IEMOCAP feature bank or GCNet hyperparameters.

## Locked protocol

- Datasets: `IEMOCAPSix` and `IEMOCAPFour`, fold 5;
- seeds: 66--70, 100 epochs, batch size 32;
- frozen features: wav2vec-large-c, DeBERTa-large-4, and MANet;
- current model: Slot observed-set node, dual-gate Top-2/4-expert MMoE, EMA teacher;
- training: all eight missing-rate views per source batch;
- selection: validation mean W-F1 over rates 0.0--0.7;
- testing: the selected checkpoint is evaluated on all eight fixed masks;
- GCNet: both temporal/speaker branches, hidden 200, window 2/2, no time attention;
- optimization: Adam `lr=1e-3`, weight decay `1e-5`, gradient clipping 1.0;
- JEPA: weight 0.1, temperature 0.03, EMA tau 0.996;
- no reconstruction, classification completion, PCIR, alternative readout, or
  Soft-Ordinal task head;
- Original and previous Missing-M3 results are inherited and are not retrained.

The only intended model/protocol delta from the previous IEMOCAP Missing-M3 runs is
the current Slot + all-rates-per-batch path. MOSI's dataset-specific `lr=5e-4` is not
transferred to IEMOCAP.

## Execution

- Remote code root: `/data2/yb/paper/GCNet_TPAMI_single_view_dev`;
- Remote result root:
  `/data2/yb/remote_experiments/missing_m3_iemocap_current_20260831/formal`;
- healthy GPUs: 2, 3, and 7; at most three jobs per GPU; GPU 4 is forbidden;
- runner: `scripts/run_missing_m3_iemocap_current.py`.

## Status

`COMPLETE` — 10/10 jobs finished with no runner failure.

## IEMOCAP-6 results

All values are test weighted F1 percentages. `Old` is the strictly mask-paired
early Missing-M3 result; `Delta` is Current minus Old.

| Miss | S66 | S67 | S68 | S69 | S70 | Current mean ± SD | Old | Delta | Positive seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.57 | 67.15 | 67.34 | 66.01 | 67.26 | 66.87 ± 0.57 | 65.01 | +1.86 | 4/5 |
| 0.1 | 66.21 | 66.94 | 66.73 | 65.48 | 68.47 | 66.77 ± 1.11 | 64.85 | +1.91 | 4/5 |
| 0.2 | 65.37 | 65.56 | 66.78 | 65.45 | 64.85 | 65.60 ± 0.71 | 64.22 | +1.39 | 3/5 |
| 0.3 | 64.58 | 66.57 | 66.43 | 64.87 | 66.27 | 65.74 ± 0.94 | 63.38 | +2.36 | 5/5 |
| 0.4 | 64.05 | 65.20 | 63.68 | 62.55 | 60.04 | 63.10 ± 1.96 | 62.43 | +0.67 | 3/5 |
| 0.5 | 63.52 | 63.81 | 62.18 | 62.52 | 62.96 | 63.00 ± 0.68 | 61.21 | +1.79 | 5/5 |
| 0.6 | 61.69 | 63.22 | 60.07 | 61.85 | 58.23 | 61.01 ± 1.91 | 60.40 | +0.61 | 4/5 |
| 0.7 | 60.95 | 61.73 | 61.79 | 61.09 | 58.37 | 60.79 ± 1.40 | 59.04 | +1.74 | 5/5 |

Best epochs are 65, 61, 68, 73, and 76 for seeds 66--70.

Aggregate comparison:

| Aggregate | Current | Old | Delta | Positive seeds |
|---|---:|---:|---:|---:|
| Miss0 | 66.865 | 65.007 | +1.858 | 4/5 |
| Eight-rate mean | 64.109 | 62.568 | +1.541 | 4/5 |
| High missing 0.4--0.7 | 61.974 | 60.771 | +1.203 | 4/5 |

The current model is positive at every rate mean. The eight-rate per-seed deltas
are `+1.386,+3.113,+3.044,+0.171,-0.009` points.

## IEMOCAP-4 results

| Miss | S66 | S67 | S68 | S69 | S70 | Current mean ± SD | Old | Delta | Positive seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 84.87 | 87.31 | 85.17 | 84.68 | 84.51 | 85.31 ± 1.15 | 85.26 | +0.05 | 2/5 |
| 0.1 | 84.49 | 87.32 | 84.91 | 83.90 | 84.75 | 85.07 ± 1.31 | 85.27 | -0.19 | 1/5 |
| 0.2 | 84.56 | 86.44 | 84.82 | 83.73 | 84.16 | 84.74 ± 1.04 | 84.47 | +0.27 | 2/5 |
| 0.3 | 81.10 | 85.59 | 84.13 | 84.29 | 83.16 | 83.65 ± 1.67 | 84.03 | -0.38 | 2/5 |
| 0.4 | 82.45 | 84.66 | 83.92 | 82.73 | 83.89 | 83.53 ± 0.92 | 83.44 | +0.09 | 3/5 |
| 0.5 | 82.10 | 83.75 | 83.51 | 82.94 | 82.78 | 83.02 ± 0.65 | 82.46 | +0.56 | 3/5 |
| 0.6 | 83.33 | 84.48 | 81.66 | 81.72 | 82.06 | 82.65 ± 1.22 | 81.88 | +0.77 | 3/5 |
| 0.7 | 82.03 | 82.05 | 81.71 | 80.25 | 81.06 | 81.42 ± 0.77 | 81.60 | -0.18 | 2/5 |

Best epochs are 91, 59, 71, 74, and 89 for seeds 66--70.

| Aggregate | Current | Old | Delta | Positive seeds |
|---|---:|---:|---:|---:|
| Miss0 | 85.307 | 85.262 | +0.046 | 2/5 |
| Eight-rate mean | 83.675 | 83.552 | +0.122 | 2/5 |
| High missing 0.4--0.7 | 82.655 | 82.347 | +0.309 | 3/5 |

The IEMOCAP-4 eight-rate per-seed deltas are
`-0.212,+1.781,-0.756,+0.567,-0.769`. This is an overall tie rather than a
stable improvement.

## Integrity audit

- 10/10 jobs returned zero and each contains exactly 100 history records;
- all history values are finite;
- 80/80 prediction NPZ files independently reproduce stored weighted F1,
  macro F1, and accuracy to below `1e-12` error;
- 80/80 predictions contain every expected class;
- 80/80 stored per-rate mask hashes are internally consistent;
- 80/80 Current mask hashes equal the same dataset/seed/rate Old hashes;
- the Current and Old NPZ files serialize availability in different flattening
  orders, so raw NPZ availability byte equality is not used as a pairing test;
- no checkpoint was copied into Git.

Current parameter counts are 32,212,238 for IEMOCAP-6 and 32,211,236 for
IEMOCAP-4, approximately 0.62% above the early mean-fusion model.

## Conclusion

The current Slot + all-rates-per-batch Missing-M3 path is a clear improvement on
IEMOCAP-6: `+1.541` eight-rate W-F1 and `+1.203` high-missing W-F1, with 4/5
positive seeds. IEMOCAP-4 is effectively unchanged overall (`+0.122`, only 2/5
positive seeds), although its high-missing aggregate is mildly positive.

Because this rerun jointly changes observed-set construction and rate scheduling,
the gain cannot be attributed to Slot or all-rates training individually. It does
show that the current Missing-M3 model did not generally regress after the MOSI-era
changes, and that the harder six-class task benefits materially.
