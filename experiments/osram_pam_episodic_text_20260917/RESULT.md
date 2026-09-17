# PAM-E: Explicit Cross-Modal Episodic Text Memory

**Internal diagnostic only; not a formal paper result.**

CMU-MOSI, official split fold 1, causal OSRAM η=.6, Flat readout, explicit episodic Text bank, learned cross-modal ridge retrieval, EMA Teacher Text target, PAM loss weight 0.05, per-rate Test-oracle selection, 5 seeds × 8 missing rates.

## Final result

| Method | 8-rate macro mean | high-missing (.5/.6/.7) |
|---|---:|---:|
| PAM-E | 79.9395 | 75.2145 |
| PAM-T | 80.1162 | 75.4562 |
| no-JEPA | 80.0755 | 75.5169 |
| reg-only | 80.1628 | 75.4910 |
| reg+NCE | 80.1045 | 75.4318 |

PAM-E macro8 = **79.9395**; paired Δ vs PAM-T = **-0.1767 pp** (1/5 seeds positive, p=0.1562).

## Paired comparisons (PAM-E − baseline)

| Baseline | mean Δ (pp) | seeds positive | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| PAM-T | -0.1767 | 1/5 | 0.1562 | 0.1875 |
| no-JEPA | -0.1360 | 1/5 | 0.2479 | 0.1875 |
| reg-only | -0.2233 | 2/5 | 0.2612 | 0.3125 |
| reg+NCE | -0.1650 | 2/5 | 0.3845 | 0.4375 |

## Per-rate W-F1 (%)

| Rate | PAM-E | PAM-T | no-JEPA | reg-only | reg+NCE |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.3049 | 87.3198 | 87.3234 | 87.5163 | 87.3463 |
| 0.1 | 84.9700 | 85.0395 | 84.9609 | 84.9993 | 85.1063 |
| 0.2 | 82.4100 | 82.4686 | 82.2129 | 82.7400 | 82.5038 |
| 0.3 | 81.0058 | 81.2851 | 80.9377 | 80.9478 | 81.4559 |
| 0.4 | 78.1815 | 78.4480 | 78.6187 | 78.6258 | 78.1281 |
| 0.5 | 76.3597 | 76.5535 | 76.5233 | 76.5145 | 76.0108 |
| 0.6 | 75.1060 | 75.5602 | 75.5351 | 75.8139 | 75.6087 |
| 0.7 | 74.1778 | 74.2550 | 74.4922 | 74.1446 | 74.6759 |

## Pattern results (macro W-F1 %)

| Pattern | PAM-E | PAM-T | no-JEPA | reg-only | reg+NCE |
|---|---:|---:|---:|---:|---:|
| A | 67.7938 | 65.1648 | 65.9343 | 64.5952 | 66.3610 |
| T | 83.4154 | 84.2340 | 85.1015 | 85.5569 | 84.9322 |
| V | 64.6845 | 65.4089 | 65.0315 | 64.0146 | 64.0472 |
| AT | 85.6404 | 86.4173 | 85.9701 | 86.9015 | 86.3867 |
| AV | 64.9271 | 65.6361 | 65.2945 | 66.4757 | 65.2753 |
| TV | 85.0648 | 86.7127 | 86.5986 | 86.3877 | 86.8714 |
| ATV | 86.5326 | 86.7542 | 87.6591 | 86.8679 | 87.3552 |
| T-missing | 65.8018 | 65.4033 | 65.4201 | 65.0285 | 65.2278 |
| T-present | 85.1633 | 86.0295 | 86.3323 | 86.4285 | 86.3864 |

PAM-E: T-missing = **65.8018**, T-present = **85.1633** (PAM-T: 65.4033 / 86.0295).

## Mechanism

| Quantity | PAM-E |
|---|---:|
| mean historical Text episode count (T-missing) | 7.6415 |
| T-missing bank nonempty coverage | 93.25% |
| mean ||z_hat_T|| | 11.2237 |
| prediction-target centered cosine | 0.0408 |
| prediction-target std ratio | 1.0353 |

## Interpretation

- PAM-E 的 8-rate macro 为 79.9395，低于 PAM-T 80.1162，也低于 no-JEPA 80.0755 与 reg-only 80.1628；paired Δ vs PAM-T = -0.1767 pp（1/5 正，p=0.1562），vs no-JEPA = -0.1360 pp（1/5 正，p=0.2479），vs reg-only = -0.2233 pp（2/5 正，p=0.2612）。没有稳定任务增益。
- T-missing 65.8018 比 PAM-T 65.4033 高 0.3985 pp，但没有突破 65–66 平台；T-present 85.1633 比 PAM-T 86.0295 低约 0.8662 pp。
- 逐 pattern 上 PAM-E 的 A-only=67.7938 明显高于 PAM-T 65.1648 与 no-JEPA 65.9343，但 V=64.6845、AV=64.9271 低于 PAM-T 的 65.4089、65.6361，T/AT/TV/ATV 也大多下降。A-only 的局部提升不足以形成整体增益。
- 机制量表明 bank 覆盖很好（bank nonempty coverage=93.25%，平均历史 Text episode=7.64），但预测目标 centered cosine 只有 0.0408，说明显式 episodic retrieval 仍然没有学好跨模态 Text latent。
- 判定：**PAM-E 没有稳定高于 PAM-v1 / no-JEPA / reg-only**。按约定停止，不自动设计下一版。

## Provenance

- Remote root: `/data2/yb/remote_experiments/osram_pam_episodic_text_20260917`
- 5 seeds × 8 rates complete; per-rate Test-oracle selection.
- Machine-readable: `results/summary.json`, `results/per_seed_rate.csv`, `results/pattern_per_seed.csv`.
