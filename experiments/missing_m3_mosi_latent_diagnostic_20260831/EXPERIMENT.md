# CMU-MOSI Missing-M3 样本级 Latent 补全诊断

## 研究问题

检查 Missing-M3 Predictor 是否根据当前 utterance 预测了真实缺失模态的样本级 EMA
latent，还是只输出目标模态的平均原型。

当前正式模型没有在推理时恢复或回灌缺失模态。Predictor 仅在训练期存在，并包含两个
独立输出：

- regression prediction：接受 SmoothL1 监督；若开启 classification completion，实际
  回灌的也是这一支；
- contrastive prediction：接受 target-specific symmetric InfoNCE 监督，不用于补全回灌。

因此本诊断分别检查两支，不能用 contrastive head 的结果替代 regression completion。

## 协议

- 数据集：CMU-MOSI；
- 模型：正式 Slot Missing-M3，learning rate `5e-4`；
- checkpoints：validation-selected seeds 66--70，直接继承，不重新训练；
- split：固定 test mask bank；
- 主诊断 missing rate：0.7；
- seed 66 额外检查 0.4、0.5、0.6、0.7；
- target shuffle：每个 target modality 内独立执行 8 次固定随机置换；
- 诊断量：Real-vs-Shuffle loss/cosine、mean-centered cosine、retrieval top-1、
  prediction/teacher effective rank 和跨样本标准差。

所有结果只用于分析表示，不参与 checkpoint 或超参数选择。

## 五种子 miss 0.7 聚合结果

### Regression completion head

| Target | Centered cosine | Real-Shuffle cosine | Real-Shuffle NCE | Retrieval | Chance | Pred rank | Teacher rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio | 0.0217 | 0.00065 | 0.0216 | 0.296% | 0.246% | 9.38 | 22.44 |
| Text | 0.0105 | 0.00032 | 0.0106 | 0.256% | 0.249% | 7.15 | 47.79 |
| Visual | 0.0169 | 0.00146 | 0.0487 | 0.304% | 0.255% | 7.30 | 18.83 |

SmoothL1 的 shuffled-minus-real 改善也只有：

- Audio：`0.00110`；
- Text：`0.00057`；
- Visual：`0.00110`。

Regression prediction 的 mean channel standard deviation 为
`0.092/0.094/0.098`，teacher 为 `0.479/0.476/0.627`。Prediction 只有 teacher
约六分之一的跨样本变化。

结论：regression head 的 retrieval 与随机机会接近，去均值后的对应关系接近零，且
effective rank 只有 `7--9 / 256`。它主要学习了 target-modality prototype，而不是
utterance-specific missing latent。

### Contrastive head

| Target | Centered cosine | Real-Shuffle cosine | Real-Shuffle NCE | Retrieval | Chance | Pred rank |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 0.0406 | 0.01335 | 0.4450 | 0.397% | 0.246% | 11.23 |
| Text | 0.0181 | 0.00386 | 0.1286 | 0.446% | 0.249% | 7.19 |
| Visual | 0.0322 | 0.01508 | 0.5026 | 0.407% | 0.255% | 10.28 |

Contrastive head 的 Real-vs-Shuffle gap 大于 regression head，说明它包含少量样本级
信息；但绝对 retrieval 仍低于 `0.5%`，effective rank 仍远低于 256，因此只能称为
弱样本信息，不能称为有效模态补全。

## 为什么原始 cosine 看起来很高

seed 66 的 regression raw cosine 在 miss 0.7 为：

- Audio：`0.8691`；
- Text：`0.9006`；
- Visual：`0.7423`。

但 target shuffle 后分别仍为约 `0.8687/0.9002/0.7420`。高 cosine 来自 teacher
latent 的共同模态方向和预测的目标模态平均原型，而不是正确的样本配对。评价补全时
必须同时报告 shuffled target 或 mean-centered metric，不能单独报告 raw cosine。

seed 66 在 missing rates 0.4--0.7 的 regression Real-vs-Shuffle cosine gap 始终小于
`0.0015`，因此该问题并非只在 miss 0.7 出现。

## 根因定位

证据支持以下数据流断点：

1. EMA teacher 仍保留更多变化，effective rank 为 `18.8--47.8`，并非完全常量；
2. regression prediction 的 rank 和 variance 显著低于 teacher；
3. SmoothL1 对高维、带强共同模态方向的 target，允许预测 target prototype 获得较低
   loss；
4. InfoNCE 使用另一套 `cl_predictions`，其少量样本级信息不会直接约束或修复
   `reg_predictions`；
5. `MissingLatentResidualFusion` 使用的是 `reg_predictions`，因此此前回灌实验收到的是
   已原型化的补全表示。

这被定义为：

`regression completion prototype collapse`

它不同于 emotion classifier 的类别坍塌，也不同于整个 EMA teacher 坍塌。

## 对当前正式结果的含义

- 当前正式配置 `classification_completion=false`，测试阶段不调用 Predictor，因此该问题
  不会直接把错误 latent 写回分类输入；
- 但 JEPA 辅助目标没有实现预期的样本级缺失模态预测，论文不能声称已完成有效
  missing-modality latent reconstruction；
- 当前 JEPA 的实际作用更接近弱的 target-modality prior regularization；
- 这能解释为什么保留 Predictor 并在测试时回灌没有带来提升。

## 结论

`COMPLETE — FAILURE MECHANISM CONFIRMED`。

在继续增加上游、图模块或 checkpoint 策略之前，必须先修复 regression completion 的
样本级辨识能力。任何后续候选都必须先通过 Real-target 明显优于 Shuffle、prediction
effective rank 恢复和 retrieval 高于随机机会的门槛，再运行情绪分类正式实验。

## 后续闭环：Joint Completion Objective

`experiments/missing_m3_mosi_joint_completion_20260831` 将 InfoNCE 从独立
`cl_predictions` 接到实际的 `reg_predictions`。五种子 paired audit 证实 Audio/Text/
Visual 的 Real-Shuffle cosine gap 均显著增加，说明 prototype shortcut 可以被直接缓解。

但相同实验的八率 emotion W-F1 从 `78.0837%` 降至 `77.2784%`。因此最终结论不是
“补全无法学习”，而是：

> actual completion 可以获得更多样本级信息，但当前共享 JEPA 梯度与 emotion
> classification 不兼容。

这关闭了“只要把 InfoNCE 接到实际补全输出即可提升分类”的路线；后续不能再次把独立
contrastive head 的好转当作 completion 已被修复，也不能用 seed 66 的中性分类结果替代
五种子结论。
