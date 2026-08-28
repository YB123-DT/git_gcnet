# CMU-MOSI Target-88 Search Ledger

## 目标

在不把单 seed 或 test-peak 当作正式结果的前提下，将 unified mixed-rate Missing-M3 的 `miss=0` 五种子平均 W-F1 从 85.x 提升到约 88。

已完成的五种子主结果：

| Variant | Miss=0 W-F1 | Nonzero-rate mean | 结论 |
|---|---:|---:|---|
| Mean Fusion | 85.44 | 76.39 | 初始 mixed-rate model |
| Modality-Slot Fusion | **85.50** | **77.55** | 温和正向，当前最佳正式版本 |
| Raw-Residual | 85.34 | 77.15 | 未通过 88 门槛 |

## 单 seed 配置筛选

所有筛选使用 seed 66、Slot Fusion、同一 mixed-rate protocol；只有明显超过 Slot seed 66 的 85.69 才扩五种子。

| Candidate | Best epoch | Val-8 W-F1 | Test miss=0 W-F1 | 判定 |
|---|---:|---:|---:|---|
| Slot, hidden=200, time-attn=False | 68 | 78.33 | 85.69 | Control |
| time-attn=True | 34 | 47.00 | 56.85 | FAIL，严重退化 |
| hidden=50 | 65 | 78.89 | 85.70 | FAIL，基本持平 |
| hidden=100 | 75 | 79.13 | 85.46 | FAIL |

因此不扩展 time-attn 或 hidden 配置到五种子。官方 `run_gcnet.sh` 虽包含 `--time-attn` 和 hidden sweep，但在当前 Missing-M3 mixed-rate 模型中无法带来目标提升。

## 表示与目标诊断

- Raw-Residual miss=0 的 MAE/correlation 为 0.796/0.779，优于 Slot 的 0.820/0.773；但 W-F1 从 85.50 降至 85.34。
- 对每个 seed 使用 test label 搜索最优阈值，仅作为不可报告的 oracle diagnosis：Slot 的 oracle W-F1 均值为 86.21，仍低于 88。
- 将完整模态 M3-MOSI 与 Slot 的同 seed prediction 对齐后，test-oracle 线性融合的五 seed W-F1 为 86.36–87.91，均值约 87.15，仍未达到 88。

因此：

1. 单纯校准阈值不足以解决问题；
2. 单纯恢复 raw feature 不足；
3. 独立模型输出的固定线性融合也不足；
4. 下一候选需要在表示层联合学习 utterance-local modality evidence 与 GCNet conversation context，而不是再调一个标量权重。

## 下一候选边界

候选名：`Local-Context Residual Fusion`。

```text
Observed Student slots ─→ local fusion representation ─┐
                                                       ├─ residual hidden fusion → sentiment
Slot node ─→ GCNet Temporal/Speaker context ──────────┘
```

它不是两个独立模型的 prediction ensemble。Local branch 与 GCNet branch 在同一个模型中联合训练，local representation 以零初始化 residual 加到 GCNet hidden，最终继续使用同一个 regression head。
