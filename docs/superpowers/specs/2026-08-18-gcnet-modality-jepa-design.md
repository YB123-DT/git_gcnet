# GCNet Missing-Modality Sweep 与 Modality-JEPA 设计规格

## 1. 目标与范围

本轮固定 `seed=66`，完成两组 IEMOCAP-Six 五折实验：

1. 复现原始 GCNet 在 `missing_rate ∈ {0.0, 0.1, ..., 0.7}` 下的结果；
2. 在相同 GCNet 分类与重建路径上增加训练期 Modality-JEPA 辅助分支，并运行同一组 missing-rate sweep。

本轮只判断“用对话上下文预测当前话轮被遮蔽模态的固定 latent，是否提升缺失模态 ERC”。不加入 EMA、未来话轮预测、多假设头、InfoNCE、MMoE、RoBERTa/DeBERTa 微调或额外上下文编码器。

## 2. 对照边界与目录隔离

原始基线代码与新方法代码必须物理隔离：

```text
GCNet_TPAMI/
├── gcnet/                         # 原始 GCNet，仅允许 NumPy 兼容修复
├── gcnet_modality_jepa/           # Modality-JEPA 独立实现
└── experiments/
    ├── original_missing_sweep_seed66_20260818/
    │   ├── miss_0p0/
    │   ├── ...
    │   └── miss_0p7/
    └── modality_jepa_seed66_20260818/
        ├── miss_0p0/
        ├── ...
        └── miss_0p7/
```

两个版本共享官方特征文件，但不共享 checkpoint、日志、预测文件或汇总文件。原始代码中的 `np.int` 仅替换为行为等价的 `int`/`np.int64`，并以回归测试锁定 mask 行为；不改变模型、损失、划分和优化逻辑。

## 3. 数据与评估协议

- 数据：GCNet 官方 IEMOCAP-Six 固定 utterance-level 特征；Audio 512D、Text 1024D、Visual 1024D。
- 划分：沿用官方五折 leave-one-session-out 实现。
- 随机种子：仅 `66`。
- 缺失率：`0.0` 到 `0.7`，步长 `0.1`。
- 训练配置：与已验证复现完全一致，LSTM、hidden size 200、past/future window 2、100 epochs。
- 主指标：Weighted-F1；同时保存 Macro-F1、Accuracy、每折最佳 epoch 和样本数。
- 汇总：五折 population mean ± std。

为保证与官方实现可比，模型选择沿用上游代码当前协议。该协议对 IEMOCAP 将测试 loader 同时作为 validation loader，存在测试集参与 epoch 选择的局限；两组实验必须完全一致，并在实验记录中明确披露，不能把结果表述为严格无泄漏评估。

## 4. 原始 GCNet 基线

原始路径保持不变：

```text
masked A/T/V features
→ BiLSTM
→ temporal graph + speaker graph
→ conversation hidden h_t (500D)
├─ classifier → emotion logits
└─ reconstruction head → concatenated raw A/T/V features
```

原始分类损失与重建损失保持上游定义。`missing_rate=0.0` 时没有缺失位置，因此重建损失为零。

## 5. Modality-JEPA 架构

### 5.1 共享主干

新方法复用原始 GCNet 的 masked input、BiLSTM、图上下文编码器、分类头和 raw reconstruction head。分类输出仍只来自原始 `h_t`，JEPA 预测不在推理时融合回分类器。

### 5.2 固定目标 latent 与训练折中心化

官方预提取的单模态特征本身作为固定 target latent。对每个 fold，仅用该 fold 的训练 session、真实 utterance、未遮蔽原始特征计算模态均值：

```text
μ_A ∈ R^512, μ_T ∈ R^1024, μ_V ∈ R^1024
r_t^m = x_t^m - μ_m
```

validation/test 只复用训练折均值，禁止重新估计。target 使用 `stop-gradient`。中心化用于削弱固定特征的公共均值方向，避免 predictor 用近似常量答案获得较高 cosine。

### 5.3 三个独立 Predictor

对 GCNet conversation hidden `h_t ∈ R^500` 使用三个独立预测器：

```text
P_A: LayerNorm(500) → Linear(500,256) → GELU → Dropout(0.1) → Linear(256,512)
P_T: LayerNorm(500) → Linear(500,512) → GELU → Dropout(0.1) → Linear(512,1024)
P_V: LayerNorm(500) → Linear(500,512) → GELU → Dropout(0.1) → Linear(512,1024)
```

输出分别预测当前话轮被遮蔽模态的 centered target：

```text
P_A(h_t) ≈ sg(r_t^A)
P_T(h_t) ≈ sg(r_t^T)
P_V(h_t) ≈ sg(r_t^V)
```

三个 head 不共享输出层，不进行融合或平均。GCNet encoder 与 predictors 都由训练损失更新；官方 A/T/V 特征不更新。

### 5.4 Mask-aware cosine loss

仅对“该 utterance 中实际被 mask 的模态”计算辅助损失。对每个模态先在有效缺失位置上平均 cosine distance，再对本 batch 中存在有效样本的模态平均：

```text
L_m = mean[1 - cosine(normalize(P_m(h_t)), normalize(r_t^m))]
L_jepa = mean({L_m | modality m has at least one masked target})
L_total = L_cls + L_rec + 0.1 * L_jepa
```

如果 batch 没有任何被遮蔽模态，`L_jepa` 返回与模型设备和 dtype 一致的有限标量零，不产生 NaN。`missing_rate=0.0` 时 JEPA 分支没有监督，结果应在随机性允许范围内退化为原始基线；它是实现一致性检查，不是方法增益点。

## 6. 诊断指标

按 Audio/Text/Visual 分别记录，仅统计真实被遮蔽的位置：

- Real cosine：预测与正确 centered target 的 cosine；
- Shuffled cosine：在同一评估集合内打乱 target 对应关系后的 cosine；
- Real–Shuffle gap；
- predictor 每维平均标准差；
- centered target 每维平均标准差；
- predictor effective rank；
- centered target effective rank；
- 有效预测样本数。

不报告同模态 Copy baseline：被 mask 的当前模态并未提供给模型，没有合法的 `x_t^m` 可复制。用未遮蔽真值计算 Copy 会引入不可用信息。诊断成功的最低证据是 Real > Shuffle 且 predictor 不是近常量；分类有效性则必须由相同 missing rate 下的五折 Weighted-F1 对照决定。

## 7. 测试与失败保护

实现前先写并观察以下测试失败，再写最小实现：

1. `random_mask` 在 NumPy 1.24+ 可运行，输出形状正确，且每个 utterance 至少保留一个模态；
2. fold mean 只读取训练 session，padding 不参与统计；
3. JEPA loss 只选择被 mask 模态；未 mask target 改变时 loss 不变；
4. 没有缺失目标时 JEPA loss 为有限零；
5. 三个 predictor 输出分别为 `[N,512]`、`[N,1024]`、`[N,1024]`；
6. stop-gradient target 不产生梯度，encoder 与 predictor 能获得梯度；
7. baseline 与 JEPA 的 fold、seed、missing rate、epoch 配置一致；
8. 小规模 smoke run 能完成训练、评估、保存 checkpoint 与诊断结果。

运行完整 sweep 前，先执行 `missing_rate=0.1` 的单折短 smoke test；通过后再启动全部五折。

## 8. 实验执行与资源约束

只占用少量 GPU：默认一张卡串行执行；如需同时推进两个互不写共享文件的队列，最多使用两张卡，分别运行 original 和 JEPA。每个实验具有独立 `GCNET_SAVED_ROOT`、stdout/stderr 日志、PID 和命令记录。不得使用或终止不属于本项目的 GPU 进程。

每个实验目录保存：

- `COMMAND.md`：完整命令、commit、环境和 seed；
- `logs/`：逐 fold 日志；
- `saved/`：checkpoint 和预测结果；
- `fold_results.json`：逐折分类与 JEPA 指标；
- `summary.json`：五折 mean ± std；
- `EXPERIMENT.md`：状态、运行时间、失败与重试记录、最终结论。

## 9. 成功与停止标准

实现成功要求：测试通过、两组 sweep 的目录和输出互不覆盖、每个 missing rate 都有五折结果、汇总可由保存预测重新计算。

研究结论按以下证据约束：

- 若 Real ≤ Shuffle 或 predictor effective rank 接近 1，则辅助任务未学到条件性模态恢复；
- 若 Real > Shuffle 但 Weighted-F1 无稳定提升，则只说明恢复任务可学，不说明有助于 ERC；
- 只有多个非零 missing rate 的 Weighted-F1 相对原始 GCNet 呈一致正向变化，才能支持“Modality-JEPA 改善缺失模态 ERC”的初步主张；
- 单 seed 结果只用于首轮筛选，不用于最终论文显著性结论。

