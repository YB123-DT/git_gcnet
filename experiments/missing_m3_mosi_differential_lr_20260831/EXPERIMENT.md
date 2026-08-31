# MOSI Differential-LR JEPA Transfer

## 假设

上一轮 joint completion 使用统一 `5e-4` 学习率，可能快速覆盖 Stage 1 JEPA 表示。该实验只改变 optimizer：预训练 Student/GCNet/Predictor 使用 `5e-5`，新 completion projection 和 emotion readout 使用 `5e-4`。

Stage 1 checkpoint、joint emotion+JEPA loss、JEPA weight 0.1、EMA、cyclic mask、features、窗口和 seed 均保持不变。

## 参数组

| 参数组 | 参数量 | 学习率 |
|---|---:|---:|
| Pretrained Student/GCNet/Predictor | 31,229,072 | 5e-5 |
| Fresh completion/readout | 387,537 | 5e-4 |

Teacher 不进入 optimizer，继续通过 EMA 更新。

## 结果

最佳 validation epoch 为 82，validation 8-rate mean W-F1 为 61.16%。

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Differential LR | 59.18 | 64.69 | 66.65 | 69.70 | 70.00 | 70.00 | 69.70 | 70.91 | 67.60 |
| Full-LR joint completion | 84.18 | 82.79 | 76.96 | 74.24 | 72.52 | 71.89 | 69.50 | 68.02 | 75.01 |
| Cyclic control | 84.92 | 84.13 | 81.18 | 79.93 | 78.00 | 75.83 | 73.73 | 74.39 | 79.01 |

Differential LR 相对 control 平均下降 11.41，且相对 full-LR joint completion 下降 7.41。

## 解释

若主要问题是快速遗忘，降低预训练参数学习率应至少优于 full-LR transfer。实际结果相反：限制主干适配后，低 missing rates 尤其严重下降，η=0 下降 25.74。

因此 Stage 2 的大更新不是单纯破坏一个已经适合情绪任务的 JEPA 表示，而是在努力把不对齐的 JEPA representation 重新塑造成 emotion representation。学习率越小，该转换越不充分。

## 决策

该实验未通过 seed-66 门槛，不扩展 seeds 67–70。关闭以下解释与操作：

- “主要因为 Stage 2 学习率过大导致 JEPA 遗忘”；
- 继续扫描 `1e-5`、`3e-5`、`1e-4` 等预训练学习率；
- 通过延长冻结期或 gradual unfreezing 修补相同 modality-latent completion 目标。

下一机制若继续，应改变 target：预测 emotion-aligned complete hidden residual，而不是继续优化当前单模态 EMA latent 的迁移策略。

