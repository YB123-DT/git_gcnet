# 混合缺失率 Missing-M3 → GCNet 设计规格

## 1. 决策

本方案将完整模态 M3-JEPA 的六方向跨模态预测迁移到缺失模态 GCNet，工作名为 **Mixed-Rate Missing-M3**。正式版本采用以下已锁定选择：

- 一个模型混合训练 `0.0–0.7` 共 8 个 missing rates，并测试全部 rates；
- JEPA Predictor 和 EMA Teacher 只在训练期存在；
- 测试时不生成、不填补、不回灌缺失模态；
- 保留 M3 的六方向 source-to-target 预测、共享 MMoE、双 gate、回归与对比双目标；
- 双 source pattern 使用两条 M3 单源预测的对称平均，不新增 attention、reliability gate 或 subset-lattice predictor；
- M3 latent prediction 替换 Original GCNet 的 raw-feature reconstruction；
- 仅保留 EMA target 和 variance floor 两项必要的低秩防护，不加入 covariance、path consistency 或多层防护组合。

该方法的核心问题是：

> 在不改变测试期 GCNet 缺失输入、不执行模态恢复的前提下，训练期预测真实缺失模态的稳定 latent，能否改善一个统一模型在不同 missing rates 下的情绪识别？

## 2. 证据与设计动机

现有完整模态 feature-level M3 实验呈现两类证据：

- CMU-MOSI：5 seeds 的 Weighted-F1 从 `86.30%` 提高到 `86.62%`，平均 `+0.32` 个百分点；
- IEMOCAP-6：无论 CSS 特征还是 GCNet 特征，M3 都出现明显低秩，且 held-out Session5 未稳定提高。

GCNet 特征上的层级诊断为：

| 表示 | Audio rank | Text rank | Visual rank |
|---|---:|---:|---:|
| 原始输入 | 43.3 | 123.7 | 11.3 |
| 随机 Projector | 52.8 | 91.0 | 28.7 |
| M3 训练后 Projector | 12.5 | 20.8 | 10.9 |

最终 Predictor effective rank 仅为 `3.61/256`。因此缺失版不能继续让分类器只依赖被 M3 压缩的 latent。本设计让分类路径保留原 GCNet observed feature，并通过零初始化 residual adapter 接收 M3 学到的增量；同时使用 EMA target 和单一 variance floor 限制 Stage-1 共同收缩。

## 3. 目标与非目标

### 3.1 目标

1. 使用官方完整训练特征构造缺失输入，并将被删除的真实模态仅作为训练 target。
2. 一个 seed 只训练一个模型，统一测试 8 个 missing rates。
3. A、T、V、AT、AV、TV 六种不完整 pattern 均有有效的 M3 prediction task。
4. Missing-M3 与无 JEPA 控制具有完全相同的测试模型与推理参数量。
5. 保持 GCNet temporal graph、speaker graph、relation、window、post-graph sequence modeling 和分类头不变。

### 3.2 非目标

本轮不实现：

- 测试时 latent completion；
- raw Audio/Text/Video Encoder 或 LoRA；
- 双视图 balanced auxiliary mask；
- source reliability gate；
- pattern attention 或 learned adjacency；
- PLCI conditional innovation；
- Original linear reconstruction；
- covariance、contrastive queue 或额外 graph layer。

## 4. 数据与混合缺失率协议

### 4.1 基本输入

对有效 utterance (i)，完整冻结特征为：

\[
X_i=[x_i^A;x_i^T;x_i^V].
\]

availability mask 为：

\[
a_i=(a_i^A,a_i^T,a_i^V)\in\{0,1\}^3,
\qquad \sum_m a_i^m\ge 1.
\]

Online Encoder 只接收：

\[
X_i^{\mathrm{miss}}=X_i\odot a_i.
\]

### 4.2 Rate 调度

训练 rates 固定为：

\[
\mathcal R=\{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7\}.
\]

每个 optimizer step 使用一个 rate。每 8 个 step 构成一个完整 cycle，8 个 rates 各出现一次；cycle 内顺序由独立、可复现的 `torch.Generator` 打乱。batch 内每个有效 utterance 按当前 rate 独立采样具体 pattern，padding 不参与采样。

这意味着：

- 同一个 batch 共享 missing rate，但具体 pattern 可以不同；
- 同一个 epoch 覆盖全部 rates，而不是让某个 epoch 只学习单一 rate；
- rate sampler RNG 与模型、DataLoader、mask bank RNG 分离；
- checkpoint 保存 rate generator state 和已经完成的 cycle/step。

### 4.3 评估

同一个训练完成的 checkpoint 分别读取 8 份固定 evaluation mask bank：

\[
F_\theta(M_{0.0}),\ldots,F_\theta(M_{0.7}).
\]

所有方法、seeds 和 rates 使用相同 mask bank。测试过程中不重新采样 mask。

## 5. 模型结构

### 5.1 Online Incomplete Encoder

每个 observed modality 使用 M3 Student Projector：

\[
s_i^m=P_m^s(\operatorname{LN}_m(x_i^m)),\qquad m\in O_i,
\]

其中：

\[
O_i=\{m:a_i^m=1\}.
\]

Student latent 通过零初始化 residual adapter 返回原 feature space：

\[
\widetilde x_i^m=a_i^m\left(x_i^m+A_m(s_i^m)\right).
\]

GCNet 输入为：

\[
\widetilde X_i=[\widetilde x_i^A;\widetilde x_i^T;\widetilde x_i^V].
\]

随后执行 Original GCNet：

\[
h_i=G_\theta(\widetilde X,Q_{\mathrm{speaker}},U_{\mathrm{valid}})_i,
\qquad o_i=C_\psi(h_i).
\]

M3 Predictor 不读取 (h_i)。这样可以保持 M3 的 source-to-target 核心，并避免 Predictor 绕过 source modality、退化成一个 fused-hidden predictor。Student Projector 仍通过 residual adapter 与分类梯度连接。

### 5.2 M3 Directional Predictor

保留当前完整模态 M3 实现中的：

- 4 个共享 experts；
- top-2 routing；
- source embedding；
- target embedding；
- regression gate 与 contrastive gate；
- target-specific regression/contrastive heads。

对单个 observed source (m) 和 missing target (q)：

\[
(p_{i,m\to q}^{\mathrm{reg}},
p_{i,m\to q}^{\mathrm{cl}})
=\operatorname{MMoE}(s_i^m,e_m,e_q).
\]

### 5.3 双 source 聚合

若 (O_i=\{m,n\}) 且 (q\notin O_i)，分别执行两条原生 M3 方向预测，再进行对称平均：

\[
\widehat p_{i,q}^{\mathrm{reg}}
=\frac12\left(
p_{i,m\to q}^{\mathrm{reg}}+
p_{i,n\to q}^{\mathrm{reg}}
\right),
\]

\[
\widehat p_{i,q}^{\mathrm{cl}}
=\frac12\left(
p_{i,m\to q}^{\mathrm{cl}}+
p_{i,n\to q}^{\mathrm{cl}}
\right).
\]

第一版不学习 source 权重。它明确检验“多个 M3 单源证据是否可以直接形成缺失 target”，避免重新引入一个独立的双源大 Predictor。

### 5.4 六种不完整 pattern

| Pattern | Missing target | Prediction |
|---|---|---|
| A | T、V | A→T；A→V |
| T | A、V | T→A；T→V |
| V | A、T | V→A；V→T |
| AT | V | mean(A→V, T→V) |
| AV | T | mean(A→T, V→T) |
| TV | A | mean(T→A, V→A) |
| ATV | 无 | 仅计算分类损失 |

单 source pattern 有 2 个 missing targets时，先在 utterance 内平均，避免 A/T/V 获得双倍权重。

### 5.5 EMA Teacher

Teacher Projector 与 Student 同构，初始化时复制 Student，始终满足：

```text
requires_grad = False
eval mode
```

对 missing target (q)：

\[
z_{i,q}^t=operatorname{stopgrad}
\left[
\mathcal N(P_q^t(\operatorname{LN}_q(x_i^q)))
\right].
\]

Teacher 只读取训练数据中被 mask 删除的真实 target feature。它不进入 GCNet、Student、Predictor 输入或分类器。

EMA 在 `optimizer.step()` 后更新：

\[
\theta_{P_q^t}\leftarrow
\tau_k\theta_{P_q^t}+(1-\tau_k)\theta_{P_q^s}.
\]

默认 momentum 从 `0.996` 余弦增加至 `0.9999`。

## 6. 损失函数

### 6.1 M3 回归损失

对每个 target (q)：

\[
\mathcal L_{\mathrm{reg}}^q
=\operatorname{SmoothL1}
(\widehat p_q^{\mathrm{reg}},z_q^t).
\]

### 6.2 M3 对比损失

保留当前 M3 实现的双向 in-batch InfoNCE：

\[
\mathcal L_{\mathrm{cl}}^q
=\frac12\left[
\operatorname{InfoNCE}(\widehat p_q^{\mathrm{cl}},z_q^t)
+\operatorname{InfoNCE}(z_q^t,\widehat p_q^{\mathrm{cl}})
\right].
\]

某个 target group 少于 2 个样本时，只计算 SmoothL1，该组 InfoNCE 为 0，不丢弃 utterance。

### 6.3 Missing-M3 损失

\[
\mathcal L_{\mathrm{M3}}=
\frac12\mathcal L_{\mathrm{reg}}
+\frac12\mathcal L_{\mathrm{cl}}.
\]

### 6.4 最小 variance floor

variance 只作用于预归一化 Student latent 和 Predictor 输出：

\[
\mathcal L_{\mathrm{var}}(Z)
=\frac1d\sum_j
\max\left(0,1-\sqrt{\operatorname{Var}(Z_{:,j})+\epsilon}\right).
\]

它不作用于 Teacher，因为 Teacher 无梯度。样本数少于 2 的 modality/target group 跳过该项。

### 6.5 总损失

主方案删除 Original `linear_rec` 及 raw reconstruction loss：

\[
\boxed{
\mathcal L=
\mathcal L_{\mathrm{cls}}
+\lambda_J\mathcal L_{\mathrm{M3}}
+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}
}
\]

默认：

```text
lambda_J = 1.0
lambda_var = 0.05
temperature = 0.03
```

本轮不同时搜索这些权重。

## 7. 训练与推理数据流

### 7.1 单个训练 step

```text
完整冻结特征 X
        │
        ├── mixed-rate mask → Xmiss
        │                     │
        │                     ▼
        │          Student Projectors + residual adapters
        │                     │
        │             ┌───────┴────────┐
        │             ▼                ▼
        │        Original GCNet    M3 Predictor
        │             │                │
        │          L_cls             p_hat_q
        │                              │
        └── missing x_q → EMA Teacher ─┘
                                   L_M3
```

顺序固定为：

1. 选择当前 batch rate；
2. 生成 availability 并构造 `Xmiss`；
3. 只用 `Xmiss` 执行 Student + GCNet；
4. 在 `torch.no_grad()` 中，从完整 `x_q` 生成 missing Teacher target；
5. 计算分类、M3 和 variance loss；
6. `backward()`；
7. `optimizer.step()`；
8. 更新 EMA Teacher；
9. 推进 rate 与 mask generator state。

### 7.2 推理

```text
固定缺失 mask 下的实际输入
→ Student Projectors + residual adapters
→ Original GCNet
→ Emotion classifier
```

推理不实例化或不调用：

- EMA Teacher forward；
- M3 MMoE Predictor；
- Missing-M3 loss；
- latent completion。

## 8. 泄漏不变量

若 (q\notin O_i)，修改完整 target (x_i^q) 不得改变：

\[
s_i^m,
\widetilde X_i,
h_i,
o_i,
\widehat p_{i,q}.
\]

它只能改变：

\[
z_{i,q}^t
\quad\text{和}\quad
\mathcal L_{\mathrm{M3}}.
\]

模型公开接口必须分离：

```python
forward_incomplete(masked_features, availability, ...)
encode_teacher_targets(full_features)
predict_missing(student_latents, availability)
update_teacher(momentum)
```

`forward_incomplete()` 不允许接收完整 features。

## 9. 代码边界

在 `feature/m3-jepa-gcnet` 分支新增：

```text
gcnet_m3_missing/
├── model.py          # Online wrapper 与 inference path
├── projectors.py     # Student/Teacher/residual adapters
├── mmoe.py           # M3 DualGateTopKMMoE
├── missing_tasks.py  # pattern → source-target task
├── rate_schedule.py  # mixed-rate cycle 与 generator state
├── loss.py           # M3 + variance loss
├── train.py          # 训练期 glue 与 EMA 顺序
└── README.md
```

复用：

- `gcnet/model.py` 的 Original graph backbone；
- `gcnet_modality_jepa` 的 split、mask bank、metrics 和 manifest 工具；
- 当前 feature-level M3 已验证的 MMoE 机制，但在本仓库进行干净重实现并记录来源。

禁止修改 Original `gcnet/` 行为来隐藏方法分支。

## 10. 对照与实验范围

第一阶段只使用 IEMOCAPSix fixed fold 5、seeds `66–70`：

| 变体 | 推理结构 | 训练目标 | 用途 |
|---|---|---|---|
| Mixed Original | Original GCNet | classification + raw reconstruction | 原方法参考 |
| Adapter-CLS | Student + adapter + GCNet | classification only | 严格推理参数匹配控制 |
| Missing-M3 | 与 Adapter-CLS 完全相同 | classification + M3 + variance | 主方案 |

每个 seed 只训练 1 个 mixed-rate checkpoint，每个 checkpoint 测试 8 个 rates。Missing-M3 通过后再扩展 CMU-MOSI；不先运行更多数据集或额外结构消融。

主判断同时报告：

- 每个 rate 的 Weighted-F1；
- 8 rates macro mean；
- 高缺失区间 `0.5–0.7` mean；
- 5 seeds 配对差值与正向 seed 数；
- 每个 pattern 和 target 的样本数、M3 loss；
- real-vs-shuffled cosine；
- Student、Teacher、prediction effective rank。

## 11. 必须通过的测试

1. 八个 rates 在一个 cycle 中各出现一次，状态可恢复。
2. 六种不完整 pattern 的 source-target 映射正确。
3. 双 source 平均对 source 顺序不敏感。
4. missing target 不进入 Student、GCNet 或 Predictor 输入。
5. 修改 missing full target 只改变 Teacher target 和 loss。
6. `optimizer.step()` 后才执行 EMA update。
7. `eta=0` 时 M3 loss 为 0，分类 forward 有效。
8. 某 target group 样本少于 2 时 InfoNCE 安全跳过。
9. padding 不参与 mask、loss、rank 或 pattern count。
10. Adapter-CLS 与 Missing-M3 的 inference state-dict keys、参数量和 forward 接口一致。
11. Predictor 与 Teacher 不进入 inference 调用图。
12. 8 个固定 evaluation mask banks 被同一 checkpoint 逐一读取。
13. CPU/GPU FP32 forward/backward 有限。
14. checkpoint 恢复后 rate cycle、mask RNG 和 EMA step 连续一致。

## 12. 可声称与不可声称

若结果通过，只能声称：

- 一个 mixed-rate 模型可覆盖多个 missing rates；
- 训练期 missing-latent prediction 改善不完整 ERC；
- M3 Predictor 不增加测试时推理成本；
- 相对于参数匹配控制，收益来自 Missing-M3 训练目标。

不能声称：

- 恢复了真实缺失模态；
- M3 原方法已经支持 dual-source missing prediction；
- EMA 或 variance 单独保证绝不坍塌；
- frozen-feature 结果等价于启用 wav2vec/DeBERTa/ViT LoRA Encoder；
- test-peak 诊断属于正式泛化结果。

## 13. 主要风险与停止条件

主要风险仍是跨模态公共子空间低秩。第一轮 smoke 必须记录 random-projector、Student、Teacher 和 Predictor rank。若 EMA + variance 下 Predictor effective rank 仍接近当前 `3–5/256`，且 real-target 与 shuffled-target 无明确间隔，则停止扩 seeds，先关闭当前 feature-level M3 目标，不通过增加更多防护继续堆叠。

若表示诊断正常但分类无提升，则说明 M3 学到了跨模态关系，但它对 GCNet missing-rate 分类没有任务价值；此时同样停止主方案，不引入 test-time completion 挽救结果。
