# CMU-MOSI Target-wise Latent Ridge Probe 设计

## 1. 目标

本实验不训练 GCNet，也不修改现有 Slot Missing-M3 checkpoint。它使用一个
target-wise、matched-capacity 的闭式 ridge residual probe，回答三个在上一轮
frozen oracle 中仍无法区分的问题：

1. EMA Teacher latent 是否包含超出 `graph_only` 的、可线性利用的情绪增量；
2. 当前部署使用的 regression prediction 是否保留了这种增量；
3. 当前只用于 InfoNCE、未进入分类 Fusion 的 contrastive prediction 是否反而
   保留了更多样本级信息。

上一轮已排除“把 prediction 直接替换为真实 Teacher，再经过现有 Fusion，就会
稳定大幅提分”。本轮专门绕开现有 Fusion 的 learned subspace，避免继续把
“latent 没信息”和“Fusion 不会读取”混为一谈。

本实验是机制诊断，不是新模型成绩，不查看测试集，也不据此对外报告 SOTA。

## 2. 已有证据与代码根因

上一轮五 checkpoint × 八 missing rates 的 validation 结果为：

- 非零 rates 的 `real_teacher - shuffled_teacher` 仅
  `+0.026 ± 0.236` 个 W-F1 百分点；
- 高缺失 rates 0.5--0.7 仅 `+0.072 ± 0.359` 个百分点；
- 高缺失下 `predicted - graph_only = -0.229 ± 0.476` 个百分点；
- Teacher 的 Fusion residual RMS 为 0.0419，Predicted 为 0.1026；
- Teacher 的 logit-shift RMS 为 0.0762，Predicted 为 0.3152；
- 两者经过 Fusion LayerNorm 后 RMS 相同，但 effective rank 和协方差方向明显
  不同，且 `tanh` 饱和率均为零。

当前代码又存在以下结构性错配：

- `reg_predictions` 由 SmoothL1 监督并进入分类 Fusion；
- `cl_predictions` 由 InfoNCE 监督，但从不直接进入分类 Fusion；
- 总损失为
  `L_cls + 0.1 × (0.5 L_reg + 0.5 L_NCE)`，所以 Teacher 回归对齐的实际系数
  只有 0.05，而分类梯度系数为 1；
- 双 source 先在 Predictor 内平均，singleton pattern 的两个 missing target 又在
  Fusion 内平均；
- 现有 Fusion 的 target-specific Linear 只在低秩 predicted manifold 上训练。

因此不能直接断言“Predictor 无信息”，也不能继续只调 JEPA 权重或替换分类损失。

## 3. 锁定范围

- 数据集：CMU-MOSI official train/validation split；
- checkpoints：上一轮相同的 seeds 66、67、68、69、70；
- missing rates：0.0--0.7，间隔 0.1；
- 模型权重：全部冻结，`eval()`，不创建 optimizer，不调用 backward；
- probe 拟合：仅使用 train split；
- probe 超参数选择：仅使用 train conversation-group cross-validation；
- 最终诊断：仅使用 validation split；
- test loader 不迭代，test 样本不用于拟合、选参、结果判读或指标汇总；
- 上游 wav2vec/DeBERTa/MANet 特征不变；
- mask schedule、graph topology、GCNet、MMoE、Teacher、classifier 和 checkpoint
  选择全部不变；
- 不重新训练 Original 或 Missing-M3；
- 不增加新依赖；
- GPU 4 禁用。

公共 loader 会初始化包含三个 split 的数据对象；准确声明是“test loader 未被
迭代或评估”，而不是“test 文件从未被加载”。

## 4. Provenance 与历史 relation mapping

每个 checkpoint 继承上一轮通过 validation history 唯一恢复的 temporal relation
row order：

| Seed | Relation row order |
|---:|---|
| 66 | `future, now, past` |
| 67 | `now, past, future` |
| 68 | `future, now, past` |
| 69 | `now, future, past` |
| 70 | `future, past, now` |

Probe runner 必须读取并校验上一轮结果中的 mapping 和 checkpoint SHA256，不能
再次依赖当前 Python 进程的 `set` 顺序。所有 train/validation 提取均在对应
mapping 下执行。

Train mask 固定使用 `epoch=0`；validation schedule 按现有协议本身固定到
`epoch=0`。这样 probe 的输入 bank 是预先定义且可复现的，不使用 checkpoint
best epoch 或 validation 结果选择 mask。

## 5. 冻结表示提取

对每个 seed、rate 和有效 utterance，收集：

- 连续 MOSI label：`y_i`；
- graph-only scalar output：`g_i = smax_fc(h_i)`；
- availability 和 target mask；
- conversation ID、utterance index、rate；
- `reg_predictions[i,q]`；
- `cl_predictions[i,q]`；
- `teacher_targets[i,q]`；
- 当前 checkpoint、mask、sample order 和 model-state SHA256。

三种 probe source 定义为：

```text
teacher    = EMA Teacher target latent
reg        = deployed regression prediction
contrastive = current InfoNCE prediction
```

只有 `target_mask[i,q] == 1` 的条目进入目标模态 `q` 的 probe pool。完整 target
只进入 no-gradient Teacher 支路；改变 missing target 的完整特征不得改变
`graph_hidden`、reg prediction 或 contrastive prediction。

Rate 0.0 没有缺失 target，不参与 ridge 拟合，只用于验证所有 probe intervention
严格退化为原 graph-only output。

## 6. Target-wise ridge residual probe

对 target modality `q ∈ {A,T,V}` 和 latent source
`s ∈ {teacher,reg,contrastive}`，定义当前 graph residual target：

```text
r_i = y_i - g_i
```

仅使用 `q` 真实缺失的 train entries，拟合：

```text
delta_hat_i^(s,q) = b_(s,q) + w_(s,q)^T standardize(z_i^(s,q))
```

最终 target-wise intervention 为：

```text
y_hat_i^(s,q) = g_i + delta_hat_i^(s,q)
```

这与当前模型的局部能力匹配：`smax_fc` 是线性头，上一轮所有 Fusion `tanh`
均未饱和，因此一个 latent-to-scalar residual ridge 足以测试现有接口附近是否存在
可线性读取的额外情绪信息。它不会用更强的深层 probe 掩盖 representation 缺陷。

### 6.1 防止八个 rate 伪扩充样本量

同一 `(conversation_id, utterance_index, target q)` 可能在多个 rates 中重复出现。
Train ridge 对该 base key 的每个 occurrence 使用：

```text
weight_i = 1 / occurrence_count(base_key)
```

因此一个原始 utterance-target 对所有 rate views 的总权重恒为 1，不把八个 mask
views 当作八个独立样本。Teacher、reg 和 contrastive 使用完全相同的条目及权重。

### 6.2 标准化与闭式解

- 仅用当前 train fold 的加权均值和标准差标准化 latent；
- 标准差下限为 `1e-6`；
- 使用 float64 CPU 求解；
- intercept 不正则化；
- ridge penalty 只作用于 `w`；
- 求解失败必须报错，不静默退回伪逆或改变 alpha grid。

Alpha grid 固定为：

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000
```

## 7. Train-only 超参数选择

每个 `(seed, target, source)` 独立使用 5-fold conversation-group CV：

1. conversation IDs 先用 SHA256 稳定排序；
2. round-robin 分配到五 folds，不使用 Python `hash()`；
3. 同一 conversation 的所有 utterances、rates 和 targets 不拆分；
4. 每个 fold 的 scaler 只在 fold-train 上拟合；
5. 以加权 residual MSE 选择 alpha；
6. 若多个 alpha 在 `1e-12` 内并列，选择更大的 alpha；
7. 选定后在完整 train pool 上重新拟合一次。

Validation 不参与 scaler、alpha、intercept 或任何参数选择。

## 8. Target-wise 与联合评估

### 8.1 Primary：单 target intervention

Audio、Text、Visual 分别评估。对于 target `q`：

- `q` 缺失的 utterance 使用 `g + delta_q`；
- 其余 utterance 保持 `g`；
- 不加入其他 missing targets 的 probe contribution。

这避免 singleton pattern 中两个目标的贡献互相抵消，直接判断哪个 target 有用。

### 8.2 Secondary：原规则联合 intervention

对一个 utterance 的缺失集合 `Q_i`：

```text
y_hat_i^joint = g_i + mean({delta_hat_i^q : q in Q_i})
```

该结果模拟当前 target averaging，但只作为次要部署接口诊断。

### 8.3 指标

每个 seed/rate/source/target 报告：

- official W-F1、Macro-F1、Acc-2；
- MAE、Pearson correlation；
- residual MSE；
- `corr(delta_hat, y-g)`；
- probe coefficient norm、selected alpha 和 effective degrees of freedom。

还报告 nonzero rates 0.1--0.7 和 high missing rates 0.5--0.7 的 seed-first
汇总。

## 9. Matched 与 shuffled 控制

对 validation 的每个 source 和 target，执行 target-wise derangement：

- 只在同 seed、同 rate、同 target 的 missing pool 内置换；
- 不跨 Audio/Text/Visual；
- 不改变 graph output、label、mask、probe head 或 latent multiset；
- pool size ≥ 2 时不允许 fixed point；
- 使用 SHA256 派生 seed；
- 初始运行 32 次；若 W-F1 Monte Carlo 标准误差大于 0.001，则扩展到 128 次；
- 保存每次 permutation SHA256 和指标。

Primary delta 为：

```text
matched source probe - mean(shuffled source probe)
```

其中 Teacher 的 matched-vs-shuffled 是判断真实目标 latent 是否包含样本级增量的
主证据；reg/contrastive 的对应 delta 判断 Predictor 输出是否保留样本级增量。

每个 seed/rate 使用 conversation-cluster bootstrap；跨 rate 汇总先在 seed 内平均，
再跨五 seeds 报告均值、population SD 和正向 seed 数。35 个 seed-rate 单元不能
被当成独立样本。

## 10. 决策门槛

一个 source 被称为具有“稳定且有实际意义的增量”，必须同时满足：

- high-missing 或 nonzero-rate 的 matched-vs-shuffled W-F1 均值至少
  `+0.25` 个百分点；
- 至少 4/5 seeds 的 seed-first delta 为正；
- 方向不能由单个 seed 独占；
- residual MSE 或 Pearson 至少一项呈现一致改善；
- 不存在 label/sample/mask 泄漏，所有结果审计通过。

`+0.25` 是诊断最低实际效应，不代表论文显著性门槛。低于它的微弱正数不足以
解释当前与目标性能之间约 2--3 个百分点的差距。

结果按下表决定后续模型，而不是看到 validation 数值后随意增加模块：

| Teacher probe | Reg probe | Contrastive probe | 结论 | 唯一下一步 |
|---|---|---|---|---|
| 正 | 弱/负 | 明显强于 Reg | 判别信息被留在未部署的 CL 分支 | 合并 reg/cl 为单一预测输出，同一输出同时承担 SmoothL1、InfoNCE 和分类 Fusion |
| 正 | 正 | 正或相近 | Latent 有用，现有 Fusion 学错子空间 | 用 target-wise ridge-like 轻量 residual adapter 替换当前 hidden-space projection |
| 正 | 弱/负 | 弱/负 | Predictor 未恢复 Teacher 中的情绪增量 | 设计低维 emotion-predictable target subspace，再预测该 subspace |
| 弱/负 | 任意 | 任意 | 当前 Teacher latent 本身没有足够线性情绪增量 | 关闭测试时 completion；JEPA 仅保留为训练期正则或正式终止该路线 |

## 11. 候选修复及取舍

### 11.1 推荐：Unified Fusion-Aligned Prediction

若 Contrastive 或 Reg probe 证明 predicted latent 中存在增量，则删除当前“两套输出、
只部署一套”的断裂：

```text
one p_q
  ├─ SmoothL1(p_q, teacher_q)
  ├─ InfoNCE(p_q, teacher_q)
  └─ target-wise residual Fusion → emotion loss
```

同一个 `p_q` 同时接收跨模态目标监督和情绪分类梯度，测试时只保留这一个输出。
它比扩大 MMoE、增加 attention 或改 BCE 更直接地解决当前证据指向的问题。

### 11.2 备选：Predictable Target Subspace

若 Teacher probe 有效而两个 Predictor probe 都无效，则不再要求 observed sources
回归 Teacher 的全部 256D 高秩细节。对每个 target 学一个小型、带防坍塌约束的
低维 target bottleneck，并让 Predictor 与 Fusion 都在该同一空间工作。该方案改动
更大，只有 probe 证明 Predictor bottleneck 后才启用。

### 11.3 安全退出：Training-only JEPA

若 Teacher probe 也无效，则直接删除测试时 completion residual。现有高缺失
`predicted - graph_only` 已为负，继续添加 reliability gate 只能学会关闭错误信号，
不能创造缺失的情绪信息。此时应保留更干净的 Online Encoder，或结束该路线。

### 11.4 明确拒绝

- 不用 BCE 替换 MOSI regression loss；这不能修复 latent 子空间；
- 不通过增大 JEPA 权重强迫不可预测的 Teacher 私有信息进入 Predictor；
- 不直接给现有 Fusion 添加 scalar reliability gate；它最多恢复 graph-only；
- 不增加 Transformer/Cross-Attention、大 MMoE 或新 GCNet 分支；
- 不训练 teacher-forced emotion head 或 logit distillation，遵守项目“不用蒸馏捷径”
  的边界；
- 不使用 validation/test 选择 ridge mask、relation order 或 checkpoint；
- 不把 privileged Teacher probe 当作可部署模型分数。

## 12. 输出与审计

独立实验目录：

```text
experiments/missing_m3_mosi_targetwise_ridge_probe_20260903/
├── EXPERIMENT.md
└── results/
    ├── summary.json
    └── seed_<seed>/
        ├── probe_models.npz
        ├── rate_<rate>.json
        └── rate_<rate>.npz
```

必须记录：

- checkpoint/config/history SHA256；
- relation mapping 与来源结果 SHA256；
- train/validation mask、sample order、labels 和 extracted latent SHA256；
- train base-key occurrence weights；
- group-fold assignment；
- scaler、alpha、ridge coefficient 与 intercept；
- matched/shuffled predictions；
- 每个 target/source 的完整指标；
- 运行前后 model-state hash 和 named-buffer 恢复状态；
- `test_split_evaluated=false`。

Runner 拒绝覆盖非空结果目录，JSON/NPZ 使用原子写入。

## 13. 必需测试

1. 加权 ridge 闭式解与一个手工小矩阵结果一致；
2. intercept 不受正则化；
3. 同 base utterance 跨 rates 的权重和为 1；
4. conversation group 不跨 CV folds；
5. scaler 只使用 fold-train；
6. alpha tie 选择更大值且完全确定；
7. Teacher/reg/contrastive 使用完全相同的 target rows；
8. target-wise intervention 不修改其他 target；
9. joint intervention 精确执行 missing-target mean；
10. shuffle 不跨 target、无 fixed point、确定性且保持 multiset；
11. 修改完整 missing target 只改变 Teacher，不改变 graph/reg/contrastive；
12. rate 0 所有 probe 输出精确等于 graph-only；
13. historical predicted path 与上一轮指标误差 `<1e-6`；
14. CPU float64 solve、GPU extraction、FP32 输入均 finite；
15. 运行前后 checkpoint state 与 named buffers 不变；
16. train-only alpha selection 和 validation-only final reporting 可由 provenance 审计；
17. NPZ 可独立重算 JSON 指标。

## 14. 执行边界

本轮只执行一个闭式 probe，不训练 100 epochs，不并行启动多个候选模型，也不重跑
Original。预计主要成本是五个冻结 checkpoint 对 train/validation 的一次表示提取；
ridge 求解本身为 CPU 小矩阵计算。

Probe 结果完成并锁定结论后，才为决策表选出的唯一模型修复另建实现规格。不得在
结果出来前同时实现 Unified Predictor、Predictable Subspace 或新 Fusion。
