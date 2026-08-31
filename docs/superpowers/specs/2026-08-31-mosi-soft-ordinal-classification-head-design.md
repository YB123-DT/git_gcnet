# CMU-MOSI 软有序分类头设计

## 研究问题

当前 CMU-MOSI 主任务使用单输出回归头与 MSE：训练拟合连续情感强度，测试时再以
零为阈值计算 Acc-2/W-F1。仓库已经完成过“双输出 + 硬标签 Cross-Entropy”的严格
配对实验；五种子结果与回归近似持平且略低，因此不能直接恢复该旧实现。

本轮只检验一个更窄的问题：能否保留连续标签中的强弱与顺序信息，同时让主任务直接
学习以零为中心的二分类边界。

## 方案选择

新增一个向后兼容的 MOSI task mode，并提供独立版本入口：

```text
--mosi-task-mode soft-ordinal
python -m gcnet_missing_m3_soft_ordinal.train_gcnet
```

模型继续输出一个 signed logit：

```text
z = Linear(h)  # [L, B, 1]
p = sigmoid(z)
```

连续标签 `y in [-3, 3]` 映射为软分类目标：

```text
t(y) = (clamp(y, -3, 3) + 3) / 6
```

训练损失为：

```text
loss = BCEWithLogits(z, t(y))
```

推理决策仍是：

```text
positive iff z > 0
negative iff z <= 0
```

因此，强正/强负样本提供更接近 `1/0` 的目标，弱情感样本靠近决策边界；`y=0`
对应 `t=0.5`，用于约束零阈值，但继续按照 MOSI Acc-2/W-F1 约定从指标中排除。

## 为什么不是旧 Binary CE

旧 binary 模式执行：

```text
y < 0 -> class 0
y > 0 -> class 1
y = 0 -> no task supervision
```

它把 `0.1` 与 `3.0`、`-0.1` 与 `-3.0` 分别压成相同标签，丢失了当前 frozen
feature bank 中仍可利用的情感强度信息。新模式仍然是以正负判别为最终任务的分类头，
但利用软目标保存标签顺序；它不是分类与回归双头，也不增加第二项主任务损失。

## 唯一变量与兼容边界

任务 contract 允许：

```text
regression    -> output_dim=1, MSE，保持原行为
binary        -> output_dim=2, hard CE，保持已完成对照
soft-ordinal  -> output_dim=1, soft BCE，本轮 treatment
```

`soft-ordinal` 与 regression 使用完全相同的 `smax_fc` 参数形状，因此参数量相同。
默认模式仍为 `regression`；旧 checkpoint、state-dict key、IEMOCAP 与 MOSEI 行为不得
改变。

## 两个版本与共享边界

保留现有入口：

```text
python -m gcnet_missing_m3.train_gcnet
```

它默认且继续代表已经验证的 regression 版本。新建：

```text
gcnet_missing_m3_soft_ordinal/
python -m gcnet_missing_m3_soft_ordinal.train_gcnet
```

它锁定 `mosi_task_mode=soft-ordinal`。两个版本共享 model、dataset、mask、Student/Teacher、
JEPA、GCNet 与训练生命周期实现；新目录只拥有版本入口、版本身份和本版本测试，不复制
整个 backbone 或训练器。这样可以分别运行、提交和归档，同时避免两份千行训练代码在
后续修复中漂移。

本轮保持不变：

- frozen wav2vec、DeBERTa 与 MANet feature bank；
- Official incomplete input、mask bank 与八个 missing rates；
- Slot Observed-Set Encoder、Student/EMA Teacher；
- Missing-M3 predictor 与 JEPA loss；
- Temporal/Speaker GCNet、图窗口与 branch fusion；
- optimizer、scheduler、epoch、fold、seed 与 checkpoint selection；
- validation 选择八个 rate 的平均 W-F1；
- 测试时不做 threshold tuning，固定使用 `z=0`。

禁止同时加入 margin loss、focal loss、class weighting、额外 MLP、回归辅助头、温度
搜索、测试集阈值校准或其他 backbone/fusion 修改。

## 代码接口

`_resolve_task_contract()` 在 CMU-MOSI 的 `soft-ordinal` 模式下返回：

```text
task = soft-ordinal
num_classes = 1
```

`_task_loss()`：

1. 只选择 `umask == 1` 的有效 utterance；
2. 对所有有效连续标签（包括 `y=0`）生成软目标；
3. 使用 `binary_cross_entropy_with_logits`；
4. 空有效 batch 返回连接 logits 的有限零值。

`_collect_predictions()` 与 `_metrics()`：

1. 指标阶段排除 `y=0`；
2. `z > 0` 映射为类别 1，否则为类别 0；
3. 计算 weighted F1、macro F1 与 accuracy；
4. 结果 artifact 保存二值 predictions、二值 labels、原 continuous labels 和 signed
   logits，使阈值与标签映射可以从 NPZ 独立重算。

## 验证要求

至少锁定以下行为：

1. 默认 regression、显式 regression 与旧 checkpoint 行为完全不变；
2. 旧 binary 模式的双 logits、hard CE 与 zero exclusion 完全不变；
3. `soft-ordinal` 输出 `[L,B,1]`，参数量与 regression 相同；
4. `y=-3,-1,0,1,3` 分别映射到 `0,1/3,1/2,2/3,1`；
5. padding 不进入 soft BCE，`y=0` 进入 loss 但不进入 Acc-2/W-F1；
6. `z=0` 是唯一固定分类阈值，不读取 validation/test 标签做阈值搜索；
7. loss、forward 与 backward 在 CPU/GPU FP32 下有限；
8. prediction NPZ 可以独立重算全部指标；
9. treatment 与 paired regression 的 test mask SHA256 完全一致；
10. feature、graph、JEPA 与 optimizer 配置逐字段相同。

## 实验协议

先在当前稳定的 CMU-MOSI Slot Missing-M3 anchor 上运行，Original/Regression 与旧
Binary 结果直接继承，不重新训练。Treatment 使用 seeds `66--70`，一个模型训练时
覆盖八个 missing rates，分别报告：

- 每个 seed、每个 missing rate 的 test W-F1；
- 五种子的逐 rate mean 与 sample std；
- miss0 mean、nonzero-rate mean 与 eight-rate mean；
- 相对 paired regression 和旧 hard-binary 的逐 seed delta；
- 最佳 epoch、两类预测计数与 signed-logit std。

判定关注稳定性而非单 seed 峰值：eight-rate mean 与 nonzero-rate mean 均需高于 paired
regression，且至少 `3/5` seeds 的 eight-rate delta 为正。失败则关闭主任务头路线，
不得继续通过阈值、类别权重或 loss 组合追分。

## 论文定位

该分类头是 task-objective alignment，不作为独立结构创新。若实验有效，它只用于说明
Missing-M3 在 MOSI 上需要一个同时保存连续标签顺序、又与最终正负判别一致的监督
接口；PLCI/Missing-M3 的方法贡献仍来自不完整表示编码与缺失 latent 预测。
