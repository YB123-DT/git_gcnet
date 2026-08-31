# Frozen JEPA Completion 设计

## 目标

验证 Stage 1 JEPA 表示本身是否包含可供 MOSI 情绪预测使用的增量信息，排除联合微调中的表示遗忘与 Predictor 输入漂移。

## 初始化与冻结边界

Stage 2 从 `jepa-only` 的固定 final checkpoint 初始化：

- 加载并冻结 `observed_set.*`；
- 加载并冻结 Temporal/Speaker GCNet 及其上下文编码参数；
- 加载并冻结 `missing_predictor.*`；
- 加载并冻结 `teacher.*`，但 Stage 2 forward 不调用 Teacher；
- 重新初始化并训练 `missing_latent_fusion.*`；
- 重新初始化并训练 `smax_fc.*`；
- 若当前 readout 类型存在额外情绪 readout，则重新初始化并训练该 readout。

Optimizer 只能接收情绪头与 completion 投影的参数。训练前后对全部冻结参数计算 SHA256，必须完全一致。

## Stage 2 数据流

```text
Official incomplete input
        ↓
Frozen Student + Frozen GCNet
        ↓
Frozen Missing-Latent Predictor
        ↓
Trainable zero-init target projection
        ↓
graph hidden + completion residual
        ↓
Trainable emotion head
        ↓
emotion loss only
```

Stage 2 不计算 JEPA、不编码 teacher target、不更新 EMA。测试路径与训练路径相同，使用真实 missing mask 决定需要补入的目标模态；η=0 时 target mask 为空，completion residual 必须严格为零。

## 实验协议

- Dataset：CMU-MOSI；
- fold：1；
- seed：66；
- train-rate mode：cyclic；
- missing rates：0.0–0.7；
- Stage 1：继承现有 seed-66 checkpoint，不重跑；
- Control：继承 cyclic seed-66 结果，不重跑；
- Stage 2：100 epochs，按 validation 8-rate mean W-F1 选择 checkpoint；
- 其余 feature、窗口、batch size 与当前 control 完全一致。

## 判定

与 cyclic control 的 seed-66 8-rate test W-F1 均值比较：

- 明确正向且高 missing rates 不恶化：再扩 seeds 67–70；
- 近似持平：说明 frozen JEPA latent 至少无害，但不足以支持主方法收益；
- 明确下降：关闭直接 latent completion 路线，不再用冻结/解冻策略调参。

## 必须验证

1. 只有 completion 投影和情绪 readout 为 `requires_grad=True`；
2. 冻结参数训练前后逐 tensor 完全一致；
3. Predictor 在 Stage 2 被调用，Teacher 与 EMA 不被调用；
4. η=0 completion residual 为零；
5. checkpoint provenance 保留 Stage 1 SHA256、epoch 与加载 key 数；
6. 结果保存 config、history、metrics，不提交 checkpoint。

