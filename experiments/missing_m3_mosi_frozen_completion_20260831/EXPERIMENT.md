# MOSI Frozen JEPA Completion 判别实验

## 问题

前一轮两阶段联合训练可能同时包含两个原因：Stage 1 JEPA latent 缺少情绪信息，或 Stage 2 微调遗忘了 Stage 1。该实验冻结全部 Stage 1 表示参数，只训练 completion projection 与 emotion readout，以隔离两者。

## 冻结边界

- 加载并冻结 Student projectors、GCNet、Missing-Latent Predictor 和 EMA Teacher；
- 重新初始化并训练 target-specific completion projections 与 `smax_fc`；
- Stage 2 只计算 emotion loss，不计算 JEPA，不调用 Teacher，不更新 EMA；
- 训练前后对全部 32,089,232 个冻结参数计算 SHA256。

Stage 1 checkpoint SHA256：`eaded043d8fd858b2b4aef331775b0c7cee5a35c26861dfc64d9386aaec96c15`。

冻结参数训练前后 SHA256 均为：

`01ca674943445c5d50e61229d8fdc9bb24eb1de74c5226f4f0e4a1757513038a`

因此结果不包含 backbone forgetting。

## 结果

最佳 validation epoch 为 58，validation 8-rate mean W-F1 为 57.08%。

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen completion | 27.22 | 46.62 | 53.27 | 56.34 | 60.00 | 62.36 | 62.98 | 62.88 | 53.96 |
| Cyclic control | 84.92 | 84.13 | 81.18 | 79.93 | 78.00 | 75.83 | 73.73 | 74.39 | 79.01 |
| Delta | -57.70 | -37.51 | -27.91 | -23.59 | -18.00 | -13.47 | -10.75 | -11.51 | -25.06 |

η=0 时 prediction standard deviation 仅 0.1845、correlation 仅 0.046，完整输入的 frozen JEPA representation 几乎不能被当前轻量情绪 readout解码。高 missing rate 分数反而较高，不表示更完整；它来自 missing-conditioned adapter/completion residual 提供的额外变化，而不是向完整情绪表示收敛。

## 结论

直接 latent completion 的失败不是由 Stage 2 遗忘造成。Stage 1 JEPA-only objective 学到了可检测的跨模态样本信息，但没有形成对 MOSI emotion regression 充分、稳定且可直接回灌的表示。

该结果未通过 seed-66 门槛，不扩展 seeds 67–70，并关闭以下路线：

- 冻结 Stage 1 后只训练 completion/readout；
- 将 JEPA similarity 或 retrieval 提升直接解释为 modality completion；
- 继续通过冻结 epoch、渐进解冻或学习率扫描修补同一 completion 公式。

