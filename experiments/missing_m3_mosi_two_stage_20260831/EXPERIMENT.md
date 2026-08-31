# MOSI 两阶段 JEPA 判别实验

## 研究问题

检验“先单独训练缺失模态 JEPA，再训练情绪模型”是否优于当前 cyclic mixed-rate control。所有比较均使用 CMU-MOSI、fold 1、seed 66、相同 frozen feature、窗口与 missing-rate 协议。

## 两阶段定义

1. `jepa-only`：100 epoch，只优化 Missing-M3 latent prediction；不计算情绪损失，不以 validation/test 情绪分数选 checkpoint，固定保存 epoch 100。
2. `emotion-only`：载入 Stage 1 的 online inference backbone，但重新初始化分类头，关闭 Predictor、Teacher 和 EMA，只训练情绪目标。
3. `joint-completion`：载入 Stage 1 的 online backbone、Predictor 与 EMA Teacher，重新初始化分类头；预测缺失 latent，经零初始化 target-specific projection 回灌 graph hidden，并联合优化情绪与 JEPA 损失。

Stage 1 checkpoint SHA256：`eaded043d8fd858b2b4aef331775b0c7cee5a35c26861dfc64d9386aaec96c15`。

## 结果

| 方法 | 最佳 epoch | 8-rate test W-F1 均值 | 相对 control |
|---|---:|---:|---:|
| Cyclic control | — | 79.01 | — |
| JEPA pretrain → emotion-only | 56 | 77.98 | -1.03 |
| JEPA pretrain → joint completion | 67 | 75.01 | -4.00 |

Joint completion 各 missing rate 的 W-F1：

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W-F1 | 84.18 | 82.79 | 76.96 | 74.24 | 72.52 | 71.89 | 69.50 | 68.02 |
| Δ control | -0.74 | -1.34 | -4.22 | -5.69 | -5.48 | -3.94 | -4.23 | -6.37 |

## 机制诊断

Stage 1 没有发生常量/低秩坍塌。η=0.7 下 regression prediction 的 centered cosine 为 A/T/V = 0.129/0.068/0.105，Real-Shuffle cosine gap 均为正，effective rank 约为 99–119/256。这证明 Predictor 含有样本级跨模态信息。

但该信息不是可直接替代完整模态的情绪充分表示：

- emotion-only 会丢弃 Predictor，预训练只作为初始化，平均下降 1.03；
- joint completion 将预测 latent 投影后加到 graph hidden，高 missing rate 下降更严重；
- η=0 时 completion residual 为零仍下降 0.74，说明联合 JEPA 训练本身也改变了 backbone 的情绪优化轨迹；
- 随缺失率增大，completion 误差累积，η=0.7 下降 6.37，反驳“latent 更接近 teacher 就必然更接近 miss0 分类表现”的假设。

## 决策

两种两阶段迁移均未通过 seed-66 判别门槛，不扩展到其余四个种子。后续若继续 completion，必须先建立“预测 latent 对情绪标签有增量信息”的条件，而不能只依赖 JEPA similarity/retrieval 指标。

