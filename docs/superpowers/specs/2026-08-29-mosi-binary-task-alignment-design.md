# CMU-MOSI 二分类任务对齐设计

## 研究问题

官方 GCNet 在 CMU-MOSI 上输出一个连续标量，并对所有有效 utterance（包含
`y=0`）优化 MSE；最终 Acc-2/W-F1 则排除 `y=0`，再按预测值是否大于零计算。
CaM-HG 的数据表将 MOSI 标记为两类，训练算法使用 Cross-Entropy。当前实验只检验
这种 task-objective mismatch 是否解释 Missing-M3/GCNet 在 MOSI 上的分数缺口。

## 唯一变量

新增配置：

```text
--mosi-task-mode regression  # 默认，严格保持已有结果
--mosi-task-mode binary      # 本轮 treatment
```

`regression` 路径必须保持：

```text
output_dim = 1
loss = MSE(prediction, continuous_label)
evaluation = remove y=0, threshold prediction at 0
```

`binary` 路径定义为：

```text
output_dim = 2
target = 0 when y < 0
target = 1 when y > 0
loss = CrossEntropy(logits, target)
prediction = argmax(logits)
```

`y=0` 与 padding 不参与 Cross-Entropy，也不进入 Acc-2/W-F1。零标签 utterance 不从
conversation 中删除：它仍进入 Observed-Set Encoder、Temporal/Speaker graph，并在
训练阶段参与与标签无关的 JEPA 表示学习。

## 保持不变

- 数据集与 standard split：1284/229/686；
- 冻结 feature bank：wav2vec、DeBERTa、MANet；
- feature dimensions 与 UID 对齐；
- natural mask schedule 和八个 missing rates；
- `train_rate_mode=all`；
- Slot observed-set fusion；
- Student/EMA Teacher、MMoE predictor 和 JEPA loss；
- Temporal/Speaker GCNet、窗口、hidden、dropout；
- optimizer、gradient clipping、100 epochs；
- eight-rate validation mean W-F1 checkpoint selection；
- test mask SHA256 与 prediction NPZ 格式。

本轮禁止加入 regression auxiliary head、联合 MSE、margin loss、class weighting、
threshold calibration、focal loss或新 graph/fusion 模块。这样任何变化只能归因于
MOSI 主任务从 regression-MSE 改为 binary-CE。

## 代码接口

`TrainConfig` 和 CLI 增加 `mosi_task_mode`，默认 `regression`。数据集 contract 在
binary 模式下返回二分类 task 与 `n_classes=2`，但不能改变 IEMOCAP、CMU-MOSEI
或现有 MOSI regression 的行为。

训练损失必须使用联合 supervision mask：

```text
supervised = valid_utterance AND (continuous_label != 0)
```

如果一个 batch 没有非零标签，分类 loss 返回连接 logits 的有限零值，不能产生
NaN，也不能跳过该 batch 的 JEPA/EMA 更新。

evaluation 收集 binary 模式的 argmax 与二值 target；保存的 `labels` 为二分类标签，
并额外保存原始连续 labels，保证审计能够确认零标签确实被排除。

## 验证

至少锁定：

1. CLI/default regression 完全向后兼容；
2. regression 模式输出、损失和 state keys 不变；
3. binary 输出为 `[L,B,2]`；
4. `y<0/y>0` 映射正确，`y=0` 与 padding 不进入 CE；
5. 修改零标签 utterance 的输入会影响 graph context，证明其没有被删除；
6. 全零标签 batch 的分类 loss 为有限零值，JEPA backward 仍有效；
7. binary prediction NPZ 独立重算 Acc-2/W-F1 一致；
8. 8/8 test mask SHA256 与 paired regression control 相同；
9. parameter count 差异只允许来自 `1 -> 2` 的 classifier output。

## 实验门槛

第一阶段只运行 CMU-MOSI、seed 66、八个 rates。paired regression control 直接继承：

```text
miss0 W-F1 = 86.56
nonzero-rate mean W-F1 = 78.31
```

扩展 seeds 67--70 的门槛：

- binary miss0 W-F1 至少 87.5，且相对 paired regression 提升至少 1.0；
- nonzero-rate mean 不低于 regression 超过 0.5；
- 无单类别坍塌，8 个 rates 的 prediction/label/mask 审计全部通过。

通过后补齐五种子，正式目标仍为 miss0 mean 约 88；失败则证明两点缺口不能仅由
task objective 解释，再回到单说话人 Speaker branch 与 MOSI temporal modeling。
