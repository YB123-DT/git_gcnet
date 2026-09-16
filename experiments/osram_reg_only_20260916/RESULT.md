# Fixed-Teacher Stage-2: regression-only vs reg+NCE vs no-JEPA

**Internal diagnostic only; not a formal paper result.**

## 结论

在固定的监督 Teacher、causal OSRAM η=.6、Flat、structured contextual MMoE、
相同 Student 初始化/数据/missing mask/训练预算和 per-rate Test-oracle 选点规则下：

| 组 | 8-rate macro mean | high-missing (.5/.6/.7) |
|---|---:|---:|
| reg+NCE | 80.1045 | 75.4318 |
| reg-only | 80.1628 | 75.4910 |
| no-JEPA | 80.0755 | 75.5169 |

三组均值：

$$\text{reg+NCE}=80.1045$$
$$\text{reg-only}=80.1628$$
$$\text{no-JEPA}=80.0755$$

最终判定：

$$\boxed{\text{reg-only 没有稳定优于另外两组}}$$

`reg-only` 虽然均值略高，但 paired 检验跨 0、3/5 seeds 为正，效应约 0.058 pp，
小于 seed 噪声。它不能声称稳定提升，只能作为“删除了无稳定收益的 InfoNCE、性能没有下降”的工程简化。

## Paired statistics

| 对比 | mean Δ (pp) | seeds | paired t p | Wilcoxon p | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| reg-only minus reg+NCE | +0.0583 | 3/5 | 0.8345 | 1.0000 | [-0.6678, +0.7845] |
| reg-only minus no-JEPA | +0.0873 | 3/5 | 0.6153 | 0.8125 | [-0.3581, +0.5326] |
| reg+NCE minus no-JEPA | +0.0290 | 3/5 | 0.8991 | 1.0000 | [-0.5660, +0.6239] |

逐 seed Δ：

| Seed | reg-only − reg+NCE | reg-only − no-JEPA |
|---:|---:|---:|
| 66 | +0.7068 | +0.1427 |
| 67 | +0.4621 | +0.5924 |
| 68 | -0.3556 | +0.2039 |
| 69 | -0.7135 | -0.3337 |
| 70 | +0.1919 | -0.1690 |

## Per-rate results (W-F1 %)

| Rate | reg+NCE | reg-only | no-JEPA |
|---:|---:|---:|---:|
| 0.0 | 87.3463 | 87.5163 | 87.3234 |
| 0.1 | 85.1063 | 84.9993 | 84.9609 |
| 0.2 | 82.5038 | 82.7400 | 82.2129 |
| 0.3 | 81.4559 | 80.9478 | 80.9377 |
| 0.4 | 78.1281 | 78.6258 | 78.6187 |
| 0.5 | 76.0108 | 76.5145 | 76.5233 |
| 0.6 | 75.6087 | 75.8139 | 75.5351 |
| 0.7 | 74.6759 | 74.1446 | 74.4922 |

## Per-seed 8-rate means (W-F1 %)

| Seed | reg+NCE | reg-only | no-JEPA |
|---:|---:|---:|---:|
| 66 | 80.2636 | 80.9704 | 80.8277 |
| 67 | 79.8917 | 80.3537 | 79.7613 |
| 68 | 80.4850 | 80.1293 | 79.9254 |
| 69 | 80.5239 | 79.8104 | 80.1441 |
| 70 | 79.3582 | 79.5501 | 79.7191 |

## Interpretation / boundary

- `reg-only` 相对 `reg+NCE`：`+0.0583 pp`，3/5 seeds 为正，paired t `p=0.8345`，Wilcoxon `p=1.0`，95% CI 跨 0。
- `reg-only` 相对 `no-JEPA`：`+0.0873 pp`，3/5 seeds 为正，paired t `p=0.6153`，Wilcoxon `p=0.8125`，95% CI 跨 0。
- `reg+NCE` 相对 `no-JEPA`：`+0.0290 pp`，3/5 seeds 为正，paired t `p=0.8991`。
- fixed-Teacher `no-JEPA` 与既有 ema `no-JEPA` 逐 seed 完全一致：因为 `emotion-only` 不使用 Teacher。
- 按预注册规则，当前属于“仍无稳定优势”：停止这一轮 loss 简化，不马上加 variance / target / predictor 来补救。
- 可以保留 `reg-only` 作为更简单的默认，但只能写“删除了无稳定收益的 InfoNCE，性能没有下降”，不能写“reg-only 提升了性能”。

## Provenance

- Dataset: CMU-MOSI, official split, fold 1.
- Selection: per-rate Test-oracle internal diagnostic.
- Teacher: supervised frozen `observed_set.projectors.*`, 100-epoch Stage-1, validation-selected.
- Code commit: `42f0aad` on `feature/osram-reg-only`.
- Raw remote roots:
  - `/data2/yb/remote_experiments/osram_supervised_teacher_20260914/student`
  - `/data2/yb/remote_experiments/osram_supervised_teacher_20260914/student-reg-only`
  - `/data2/yb/remote_experiments/osram_supervised_teacher_20260914/student-emotion-only`
- Machine-readable: `per_seed_rate.csv`, `summary.json`.

