# MOSI All-Rates-Per-Batch 训练设计

## 问题

CMU-MOSI official train split 只有 52 段 conversation。`batch_size=32` 时每个 epoch 只有 2 个 batch；现有 cyclic mixed-rate trainer 每个 batch 只使用一个 missing rate，因此 100 epochs 中每个 rate 仅得到约 25 个 batch、约 12.5 个完整训练集等效曝光。validation 却在每个 epoch 同时评估八个 rates，导致训练信号覆盖与选择目标不对称。

本实验不修改模型，而是检验该优化预算是否为 MOSI 的主要瓶颈。

## 唯一变量

新增训练模式：

```text
train_rate_mode = cyclic | all
```

- `cyclic`：保留当前行为，每个 batch 只训练一个 rate；
- `all`：每个原始 conversation batch 依次训练 `0.0...0.7` 八个 views，八个 loss 平均后只执行一次 optimizer step。

## 精确操作

对一个原始 batch：

```python
optimizer.zero_grad(set_to_none=True)
teacher = encode_teacher_targets(complete_view)

for rate in MISSING_RATES:
    view = prepare_view(raw_batch, schedule[rate], epoch)
    logits, predictions = model(view)
    loss_rate = loss_cls + lambda_jepa * loss_jepa
    (loss_rate / len(MISSING_RATES)).backward()

clip_grad_norm(...)
optimizer.step()
model.update_teacher(tau)
```

Teacher target 可以从完整特征计算一次并复用；它不读取 availability。每个 rate 的 predictor target mask 仍由该 rate 的真实 availability 决定。

## 不变量

- model architecture、parameters、initialization、loss 公式不变；
- natural mask schedule、seed、conversation IDs、epoch 参数不变；
- 每个 batch 仍只有一次 optimizer step、一次 gradient clipping 和一次 EMA update；
- validation/test、八-rate checkpoint selection 和 inference 完全不变；
- `cyclic` 为默认值，旧 checkpoint/config/训练行为不变；
- 不增加 reconstruction、第二 view、attention、fusion head 或 LoRA。

## 记录

`TrainConfig` 与 CLI 新增：

```text
--train-rate-mode cyclic|all
```

每 epoch 的 `rate_batch_counts`：

- cyclic：总和等于 `len(train_loader)`；
- all：每个 rate 都等于 `len(train_loader)`。

同时记录 `optimizer_steps`，MOSI 100 epochs 应为 200，证明 all-mode 没有增加 Adam step 数。

## 测试

1. 默认 cyclic 路径的调用、参数和输出行为不变；
2. all mode 每个 batch 恰好覆盖八个 rates；
3. 八个 loss 按 `1/8` 累积；
4. 每个 batch optimizer/EMA 只更新一次；
5. teacher complete target 每 batch只编码一次；
6. rate-specific missing target count 非零且 backward finite；
7. CLI/config/checkpoint 记录训练模式；
8. validation/test 不受训练模式影响。

## 判别实验

- Dataset：CMU-MOSI official split；
- Model：当前最佳 Slot Missing-M3；
- Seed：66；epochs：100；hidden=200；time-attn=False；
- Control：继承 cyclic seed66；
- Treatment：all-rates-per-batch；
- 扩展门槛：miss=0 W-F1 ≥87.0，且 nonzero-rate mean 相对 Slot 不低于 -0.5；
- 未过门槛不扩 seeds 67–70。

