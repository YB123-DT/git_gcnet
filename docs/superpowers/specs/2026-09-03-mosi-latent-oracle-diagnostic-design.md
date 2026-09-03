# MOSI Missing-Latent Oracle 诊断设计

## 目标

在不重新训练主模型的前提下，判断 CMU-MOSI 上的性能瓶颈更接近以下哪一种：

1. Missing-Latent Predictor 没有恢复足够的样本级信息；
2. 已训练的 fusion 与 Emotion Head 无法利用目标 latent。

该实验只用于机制诊断，不构成新的主方法，也不把 privileged target 结果作为正式模型成绩。

## 锁定对象

- 数据集：CMU-MOSI official split；
- 模型：`classification_completion=true` 的 Slot Missing-M3；
- 代码基线：提交 `c4874e4f8782044ebf283cf57f5b5a4e1bacb524`；
- checkpoints：远端正式实验 seeds 66--70 的 `best.pt`；
- 首轮 rates：0.4、0.7；
- 首轮 split：validation；
- mask：由 checkpoint 中的 seed 和原 `ConversationMaskSchedule` 重建；
- 权重：全部冻结，不执行 optimizer，不更新 EMA；
- shuffle：每个 seed、rate 执行 8 个固定置换。

历史基线代码必须先复现保存结果。当前 HEAD 已包含后续结构变化，不允许作为本诊断的执行基线。

## 四条推理路径

每个 batch 只计算一次 observed encoder、GCNet graph hidden、Missing-Latent Predictor 和 EMA teacher：

1. `graph_only`：直接将原始 `graph_hidden` 送入 `smax_fc`；
2. `predicted`：将原 predictor 的 `reg_predictions` 经原 `missing_latent_fusion` 加回 `graph_hidden`；
3. `real_teacher`：将当前样本真实缺失模态的 EMA teacher latent 经同一个 fusion 加回；
4. `shuffled_teacher`：在相同目标模态的缺失位置池内置换 teacher latent，再经同一个 fusion 加回。

所有路径共用：

- 同一个 checkpoint；
- 同一个 incomplete graph hidden；
- 同一个 `target_mask`；
- 同一个 classifier；
- 同一批样本顺序。

`graph_only` 必须直接绕过 fusion，不能把 latent 置零后送入 fusion，因为 fusion 的线性层 bias 可能产生非零 residual。

## 全 split 置换

先按 MOSI 指标顺序收集 validation split 的有效 utterance：

```text
[B, L, ...] --valid mask--> [N, ...]
```

对 Audio、Text、Visual 分别处理。仅在 `target_mask[:, modality] == 1` 的位置内置换 latent，不跨模态、不触碰 padding、不改变 target mask。随机种子由以下字段稳定派生：

```text
checkpoint seed + rate + shuffle index + target modality
```

不使用 Python 内置 `hash()`。

## 输出

每个 seed/rate 保存：

- 四路 W-F1、Acc-2 和回归指标；
- `real_teacher - predicted`、`real_teacher - shuffled_teacher`；
- 8 次 shuffled teacher 的均值、标准差和逐次结果；
- prediction 与原生 model forward 的最大绝对误差；
- checkpoint SHA256、mask SHA256、代码提交；
- prediction/teacher 的 target-wise channel std 与 effective rank；
- labels、availability、四路 predictions 和 8 份 shuffle predictions 的 NPZ。

根目录生成五种子均值、标准差和配对 delta 汇总。

## 必须通过的测试

1. 手动 `predicted` 路径与原生 completion forward 等价；
2. `graph_only` residual 精确为零；
3. teacher 替换只作用于真实缺失目标；
4. shuffle 不跨 target modality，且保持每个目标池的多重集合不变；
5. 相同 seed 的 shuffle 完全一致，不同 shuffle index 至少一个目标池不同；
6. padding 与 observed target 不影响结果；
7. 不执行 teacher update，运行前后 model state-dict SHA256 不变；
8. MOSI 的 flatten 顺序与 prediction/label/availability 对齐；
9. 输出均为 finite，NPZ 与 JSON 可重算指标。

## 解释边界

- `real_teacher` 使用测试时不可见的真实缺失特征，只能称为 oracle diagnostic；
- `shuffled_teacher` 是目标模态内的负对照，不是可部署基线；
- graph hidden 仍来自 incomplete input，因此 real teacher 不是完整模态性能上界；
- fusion 与 classifier 只在低秩 predicted latent 上训练，teacher latent 对它们属于分布外输入；
- 因而 real teacher 明显优于 predicted/shuffled 可支持「predictor 是瓶颈」，但 real teacher 不提升不能证明 teacher target 没有情感信息。

若直接注入结果为阴性，再单独设计冻结 supervised probe；首轮实现不把 probe 混入四路诊断。

## 停止条件

首轮在 validation 的 5 seeds × 2 rates 上完成后即停止并汇报，不自动查看 test。只有证据表明 real teacher 在 validation 上稳定优于 shuffled teacher，才讨论是否需要一次锁定后的 test 验证。
