# PAM-A: Explicit Episodic Text Memory with Direct Address Supervision

**Internal diagnostic only; not a formal paper result.**

CMU-MOSI, fold 1, causal OSRAM eta=.6, Flat readout, explicit Text bank, signed per-episode address scorer, direct oracle-address SmoothL1, EMA Teacher Text target, per-rate Test-oracle selection, 5 seeds x 8 missing rates.

## Final result

| Method | 8-rate macro mean | high-missing (.5/.6/.7) |
|---|---:|---:|
| PAM-A | 79.8194 | 75.0342 |
| PAM-E | 79.9395 | 75.2145 |
| PAM-T | 80.1162 | 75.4562 |
| no-JEPA | 80.0755 | 75.5169 |
| reg-only | 80.1628 | 75.4910 |
| reg+NCE | 80.1045 | 75.4318 |

PAM-A macro8 = **79.8194**; paired Δ vs PAM-E = **-0.1201 pp** (2/5 seeds positive, p=0.3653).

## Paired comparisons (PAM-A − baseline)

| Baseline | mean Δ (pp) | seeds positive | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| PAM-E | -0.1201 | 2/5 | 0.3653 | 0.4375 |
| PAM-T | -0.2968 | 1/5 | 0.0946 | 0.1250 |
| no-JEPA | -0.2561 | 2/5 | 0.2042 | 0.3125 |
| reg-only | -0.3434 | 2/5 | 0.2064 | 0.3125 |
| reg+NCE | -0.2850 | 1/5 | 0.3489 | 0.3125 |

## Per-rate W-F1 (%)

| Rate | PAM-A | PAM-E | PAM-T | no-JEPA | reg-only | reg+NCE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.5352 | 87.3049 | 87.3198 | 87.3234 | 87.5163 | 87.3463 |
| 0.1 | 84.9153 | 84.9700 | 85.0395 | 84.9609 | 84.9993 | 85.1063 |
| 0.2 | 82.2867 | 82.4100 | 82.4686 | 82.2129 | 82.7400 | 82.5038 |
| 0.3 | 80.5979 | 81.0058 | 81.2851 | 80.9377 | 80.9478 | 81.4559 |
| 0.4 | 78.1177 | 78.1815 | 78.4480 | 78.6187 | 78.6258 | 78.1281 |
| 0.5 | 75.9250 | 76.3597 | 76.5535 | 76.5233 | 76.5145 | 76.0108 |
| 0.6 | 75.0962 | 75.1060 | 75.5602 | 75.5351 | 75.8139 | 75.6087 |
| 0.7 | 74.0813 | 74.1778 | 74.2550 | 74.4922 | 74.1446 | 74.6759 |

## Pattern results (macro W-F1 %)

| Pattern | PAM-A | PAM-E | PAM-T | no-JEPA | reg-only | reg+NCE |
|---|---:|---:|---:|---:|---:|---:|
| A | 65.7048 | 67.7938 | 65.1648 | 65.9343 | 64.5952 | 66.3610 |
| T | 83.8663 | 83.4154 | 84.2340 | 85.1015 | 85.5569 | 84.9322 |
| V | 60.6426 | 64.6845 | 65.4089 | 65.0315 | 64.0146 | 64.0472 |
| AT | 85.8814 | 85.6404 | 86.4173 | 85.9701 | 86.9015 | 86.3867 |
| AV | 64.1179 | 64.9271 | 65.6361 | 65.2945 | 66.4757 | 65.2753 |
| TV | 85.4855 | 85.0648 | 86.7127 | 86.5986 | 86.3877 | 86.8714 |
| ATV | 86.7418 | 86.5326 | 86.7542 | 87.6591 | 86.8679 | 87.3552 |
| T-missing | 63.4884 | 65.8018 | 65.4033 | 65.4201 | 65.0285 | 65.2278 |
| T-present | 85.4938 | 85.1633 | 86.0295 | 86.3323 | 86.4285 | 86.3864 |

PAM-A: T-missing = **63.4884**, T-present = **85.4938**.

## Mechanism

| Quantity | PAM-A |
|---|---:|
| mean historical Text episode count (T-missing) | 7.6415 |
| T-missing bank nonempty coverage | 93.25% |
| mean ||z_hat_T|| | 9.0678 |
| prediction-target centered cosine | 0.0140 |
| prediction-target std ratio | 1.2289 |

## Information probe (seed 66 PAM-E checkpoint)

Direct MLP regressors from the same current-visible A/V + OSRAM Base/Gap + pattern features, trained on train split and evaluated on test split over rates .1/.3/.5/.7:

| Target space | test centered cosine | test raw cosine | test SmoothL1 | pattern-mean SmoothL1 |
|---|---:|---:|---:|---:|
| full Teacher Text 256d | 0.0886 | 0.9176 | 0.0802 | 0.0748 |
| 32d predictable subspace | 0.1182 | 0.5532 | 0.0647 | 0.0621 |

The direct probe reaches only ~0.09 centered cosine for full Text and ~0.12 for the 32d subspace; the pattern-conditioned mean is still better on SmoothL1. This indicates that the current context contains only a weak sample-specific Text signal, which is consistent with PAM-A not improving over PAM-E.

## Interpretation

- PAM-A macro8 79.8194, below PAM-E 79.9395, PAM-T 80.1162, no-JEPA 80.0755, and reg-only 80.1628.
- Paired Δ vs PAM-E = -0.1201 pp (2/5 positive, p=0.3653); vs no-JEPA = -0.2561 pp; vs reg-only = -0.3434 pp.
- T-missing 63.4884 is ~2.31 pp below PAM-E; V-only 60.6426 and AV 64.1179 drop sharply. Direct address loss did not fix the plateau.
- The information probe shows weak current-context-to-full-Text predictability; no stable task gain is achieved. The direct address-supervision route is not worth continuing as-is.

## Provenance

- Remote root: `/data2/yb/remote_experiments/osram_pam_address_text_20260918`
- 5 seeds x 8 rates; per-rate Test-oracle selection.
- Machine-readable: `results/summary.json`, `results/per_seed_rate.csv`, `results/pattern_per_seed.csv`, `results/address_probe_seed66.json`.
