# PAM-T: Target-Conditioned Predictive Associative Memory (Text target)

**Internal diagnostic only; not a formal paper result.**

CMU-MOSI, official split fold 1, causal OSRAM η=.6, Flat readout,
EMA Teacher Text target, regression-only PAM loss weight 0.05,
per-rate Test-oracle selection, 5 seeds × 8 missing rates.

## Final result

| Method | 8-rate macro mean | high-missing (.5/.6/.7) |
|---|---:|---:|
| no-JEPA | 80.0755 | 75.5169 |
| reg-only | 80.1628 | 75.4910 |
| reg+NCE | 80.1045 | 75.4318 |
| **PAM-T** | **80.1162** | **75.4562** |

PAM-T macro8:

$$\text{PAM-T}=80.1162$$

最终判定：

$$\boxed{\text{PAM-T 没有稳定优于 no-JEPA / reg-only / reg+NCE}}$$

## Paired comparisons (PAM-T − baseline)

| Baseline | mean Δ (pp) | seeds | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| no-JEPA | +0.0407 | 1/5 | 0.7431 | 0.6250 |
| reg-only | -0.0466 | 2/5 | 0.6924 | 0.8125 |
| reg+NCE | +0.0117 | 3/5 | 0.9602 | 0.8125 |

Per-seed Δ：

| Seed | PAM-T − no-JEPA | PAM-T − reg-only | PAM-T − reg+NCE |
|---:|---:|---:|---:|
| 66 | -0.1459 | -0.2886 | +0.4182 |
| 67 | +0.4980 | -0.0944 | +0.3677 |
| 68 | -0.0517 | -0.2556 | -0.6113 |
| 69 | -0.0583 | +0.2754 | -0.4381 |
| 70 | -0.0386 | +0.1303 | +0.3222 |

## Per-rate W-F1 (%)

| Rate | PAM-T | no-JEPA | reg-only | reg+NCE |
|---:|---:|---:|---:|---:|
| 0.0 | 87.3198 | 87.3234 | 87.5163 | 87.3463 |
| 0.1 | 85.0395 | 84.9609 | 84.9993 | 85.1063 |
| 0.2 | 82.4686 | 82.2129 | 82.7400 | 82.5038 |
| 0.3 | 81.2851 | 80.9377 | 80.9478 | 81.4559 |
| 0.4 | 78.4480 | 78.6187 | 78.6258 | 78.1281 |
| 0.5 | 76.5535 | 76.5233 | 76.5145 | 76.0108 |
| 0.6 | 75.5602 | 75.5351 | 75.8139 | 75.6087 |
| 0.7 | 74.2550 | 74.4922 | 74.1446 | 74.6759 |

## Pattern results (macro W-F1 %)

| Pattern | PAM-T | no-JEPA | reg-only | reg+NCE |
|---|---:|---:|---:|---:|
| A | 65.1648 | 65.9343 | 64.5952 | 66.3610 |
| T | 84.2340 | 85.1015 | 85.5569 | 84.9322 |
| V | 65.4089 | 65.0315 | 64.0146 | 64.0472 |
| AT | 86.4173 | 85.9701 | 86.9015 | 86.3867 |
| AV | 65.6361 | 65.2945 | 66.4757 | 65.2753 |
| TV | 86.7127 | 86.5986 | 86.3877 | 86.8714 |
| ATV | 86.7542 | 87.6591 | 86.8679 | 87.3552 |

PAM-T: T-missing = **65.4033**, T-present = **86.0295**.

## Interpretation

- PAM-T macro8 `80.1162`；相对 no-JEPA `+0.0407 pp`（1/5 seeds 为正，p=0.7431），相对 reg-only `-0.0466 pp`（2/5，p=0.6924），相对 reg+NCE `+0.0117 pp`（3/5，p=0.9602）。
- 三组 paired 检验全部跨 0，PAM-T 没有稳定任务增益。
- T-missing `65.40` vs T-present `86.03`，gap 约 20.6 pp；动态 A/V→Text memory 仍然没有修复缺失 Text 瓶颈。
- 逐 pattern 上，PAM-T AV=65.64，高于 no-JEPA 65.29、低于 reg-only 66.48；A/V-only 约 65.2/65.4，T-only 84.23，均未形成稳定优势。
- 原 full-run NPZ 未保存 `pam_prediction_text` / `pam_target_text`；task 与 pattern 结果完整，但 PAM centered cosine 和 std ratio 需要单独用修正后的 trainer 做 evaluation-only 才能补齐。

## Provenance

- Remote root: `/data2/yb/remote_experiments/osram_pam_text_20260917`
- Selection: per-rate Test-oracle; 5 seeds × 8 rates = 40 jobs, COMPLETE 2026-09-17T07:35:41Z.
- Machine-readable: `results/summary.json`, `results/per_seed_rate.csv`, `results/pattern_per_seed.csv`.

## Inference ablation

Four-mode inference on the same checkpoints and test masks:

| Mode | nonzero-rate macro W-F1 |
|---|---:|
| Normal | 79.0871 |
| Zero | 77.9831 |
| Shuffle | 78.0395 |
| Oracle-Fusion | 81.4504 |

- Normal − Zero = `+1.1041 pp`
- Normal − Shuffle = `+1.0476 pp`
- Oracle − Normal = `+2.3633 pp`
- prior-write coverage = `85.52%`

This is closest to **fusion can use Text, but PAM did not predict it well**:
the completion path is used and carries some sample-specific information, but
real Text latent through the same interface is much better.  See
`INFERENCE_ABLATION.md`.
