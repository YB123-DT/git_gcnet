# Missing-M3 下游 Learning Rate 筛选设计

## 1. 研究问题

现有 CMU-MOSI 下游实验全部固定使用 Adam `lr=1e-3`，尚未做过正式 learning-rate
筛选。本实验只回答：冻结上游特征后，当前 Slot Missing-M3 + GCNet 是否因下游 LR
不合适而低估 MOSI 性能。CMU-MOSEI 暂不运行。

## 2. 模型边界

保持冻结：

- wav2vec Audio feature；
- DeBERTa Text feature；
- MANet Visual feature。

本轮不训练 Transformer、不生成新特征、不使用 LoRA。可训练部分仅为现有：

```text
ObservedSetEncoder (Slot)
→ GCNet Temporal/Speaker graph
→ Missing-M3 DualGate MMoE（训练期）
→ Emotion regression head
```

## 3. 唯一变量

Control 直接继承：

```text
lr = 1e-3
```

新候选：

```text
lr ∈ {3e-4, 5e-4, 2e-3}
```

optimizer 仍为 Adam；weight decay、batch size、epochs、gradient clipping、EMA、JEPA
权重和所有模型参数不变。不新增 scheduler 或 warmup，避免把 LR 与调度策略混在一起。

## 4. CMU-MOSI 锁定协议

- official split，fold 1；
- Slot、Regression-MSE、`train_rate_mode=all`；
- 一个 checkpoint 测试八个 missing rates；
- 当前 `lr=1e-3` Control 五种子直接继承。

## 5. 分阶段实验

第一阶段只运行 MOSI seed 66：

```text
1 dataset × 3 new LR = 3 tasks
```

LR 只能按 validation 八-rate W-F1 均值选择。虽然训练器会保存 test 输出，选择
过程不得读取 test metric。

若没有新 LR 超过 `1e-3` 的 seed66 validation score，则保留 `1e-3` 并关闭 LR 路线。
若存在正向候选，则只保留 validation 最优 LR，补 seeds 67--70：

```text
1 dataset × 4 remaining seeds = 4 tasks
```

## 6. 正式判定

五种子结果需要同时报告：

- 八-rate W-F1 mean；
- miss0 W-F1；
- high-missing `0.4--0.7` mean；
- 逐 seed paired delta；
- best epoch 分布；
- 40/40 test mask SHA256 与 Control 的配对结果。

只有八-rate mean 为正、至少 3/5 seed 为正且 high-missing 不下降，才把新 LR 替换为
后续下游默认值。否则继续使用 `1e-3`。

## 7. 禁止变化

本轮不修改：模型代码、Loss、MMoE、图结构、文本特征、mask generator、missing-rate
训练模式、checkpoint selection、epoch 或推理路径。Control 不重新训练。
