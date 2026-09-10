# Completed causal eta=.6 readout ablation

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

All 15 newly trained MOSI runs completed 100 epochs with successful exits; Full is inherited. Reporting selects the maximum Test weighted-F1 independently for every seed66–70 and every rate0.0–0.7, using earliest epoch on ties. These are not eight rates from one checkpoint. The summary validated complete histories, configuration pairing and identical test mask hashes; local verification independently checked all160 selected records and aggregate means.

## Five-seed W-F1 (%)

| Rate | Full | Local-only | Local+Base | Local+Gap |
|---|---:|---:|---:|---:|
| 0.0 | 87.42 | 87.18 | 87.06 | 87.36 |
| 0.1 | 85.40 | 84.20 | 84.65 | 84.85 |
| 0.2 | 81.94 | 80.77 | 81.68 | 81.92 |
| 0.3 | 80.91 | 79.45 | 80.69 | 81.33 |
| 0.4 | 78.09 | 75.91 | 77.74 | 77.86 |
| 0.5 | 76.63 | 72.91 | 76.19 | 76.39 |
| 0.6 | 75.09 | 72.70 | 75.65 | 75.52 |
| 0.7 | 74.55 | 70.93 | 73.71 | 73.92 |
| Eight-rate mean | 80.0022 | 78.0056 | 79.6706 | 79.8934 |
| High missing (.5/.6/.7) | 75.4244 | 72.1799 | 75.1843 | 75.2768 |

| Variant | Eight-rate seed mean ± sample std | Delta to Full (pp) | Seeds above Full |
|---|---:|---:|---:|
| Full | 80.0022 ± 0.4677 | 0 | — |
| Local-only | 78.0056 ± 0.3197 | -1.9967 | 0/5 |
| Local+Base | 79.6706 ± 0.3109 | -0.3316 | 1/5 |
| Local+Gap | 79.8934 ± 0.2052 | -0.1088 | 1/5 |

## Interpretation and limits

- Removing both classification context families reduces the high-missing mean by 3.2444pp versus 0.2374pp at miss0. Context is useful in this trained architecture, especially under missing inputs.
- Base alone recovers 1.6650pp of the eight-rate deficit; Gap alone recovers 1.8878pp. Both supply useful contextual readout. The experiment does not establish that one is statistically superior.
- Full exceeds Local+Base by 0.3316pp and Local+Gap by 0.1088pp. Small incremental gains are consistent with overlapping utility, but do not prove semantic redundancy, a defective fusion mechanism, or that attention would improve F1.
- These are fresh retrained ablations. At fixed parameters the mask leaves memory trajectory and structured predictor inputs unchanged; after retraining their weights need not match. Local-only still has auxiliary JEPA learning, and is not a memory-free training system.
- Tests are used repeatedly for selection. Results are optimistic diagnostics, not held-out generalization estimates or a basis for claiming a statistically significant method improvement.

## Artifacts

- `results/per_seed_rate.csv`: all160 selected seed/rate/variant scores, epochs and mask hashes (F1 stored as fractions).
- `results/summary.json`: exact means, sample std, paired deltas and seed counts.
- `results/summary.md`: expanded tables including every selected epoch.
- `COMPLETED_QUEUE.json`: completed process and exit status evidence; `LAUNCH_SNAPSHOT.json` remains a historical launch snapshot.
- Remote raw histories/configs/metrics/provenance: `/data2/yb/remote_experiments/osram_causal_readout_20260910/mosi/{local-only,local-base,local-gap}/seed_{66..70}/`.
- Source code `9d795f0`; launch `7d173cf`; no new models or experiments started during result collection.
