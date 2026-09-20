# MOSI cyclic training without miss=0.0

This is an internal diagnostic, not a formal paper result.

## Protocol

- Model: cfg84 no-JEPA causal OSRAM, write step 0.6, Flat readout.
- Training: cyclic schedule over `0.1, 0.2, ..., 0.7`; miss=0.0 is never sampled for optimization.
- Evaluation: all eight rates `0.0, 0.1, ..., 0.7`.
- Seeds: 66–70.
- Checkpoint selection: each seed × each rate independently selects the highest Test weighted-F1 epoch (`per-rate-test-oracle`).
- Remote output: `/data2/yb/remote_experiments/osram_no_aux_cfg84_cyclic_no0_20260920/`.

## Result

| metric | cyclic original | cyclic train 0.1–0.7 | delta |
|---|---:|---:|---:|
| 8-rate mean | 80.445% | 80.038% ± 0.457% | -0.407 pp |
| 7-rate mean (0.1–0.7) | 79.357% | 79.005% ± 0.460% | -0.353 pp |
| high missing mean (0.5–0.7) | 75.650% | 75.496% ± 0.857% | -0.153 pp |

The per-seed and per-rate values are in [summary.csv](summary.csv). The raw per-seed `config.json`, `metrics.json`, and `PROVENANCE.json` files are under `results/seed_*`.

## Per-rate five-seed mean ± SD

| miss rate | cyclic original | cyclic train 0.1–0.7 | delta |
|---:|---:|---:|---:|
| 0.0 | 88.059 ± 0.541 | 87.270 ± 0.606 | -0.789 |
| 0.1 | 85.386 ± 0.929 | 85.072 ± 0.634 | -0.315 |
| 0.2 | 82.800 ± 1.280 | 81.871 ± 1.043 | -0.929 |
| 0.3 | 81.330 ± 0.882 | 80.958 ± 1.304 | -0.372 |
| 0.4 | 79.035 ± 1.315 | 78.644 ± 1.679 | -0.391 |
| 0.5 | 76.750 ± 1.315 | 76.231 ± 1.629 | -0.519 |
| 0.6 | 75.671 ± 0.526 | 76.076 ± 0.712 | +0.406 |
| 0.7 | 74.529 ± 2.955 | 74.182 ± 3.056 | -0.347 |

## Conclusion

Omitting the complete-modality (`miss=0.0`) training condition did not improve this cfg84 cyclic MOSI run. The change is therefore **No-Go** as a standalone training modification; it should not replace the original cyclic schedule.
