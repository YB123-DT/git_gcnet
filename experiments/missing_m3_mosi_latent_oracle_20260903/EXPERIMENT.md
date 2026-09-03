# CMU-MOSI Missing-Latent Oracle 诊断

> 状态：`COMPLETE / NEGATIVE DIAGNOSTIC`。正式运行代码版本为
> `34bac0e69c4f13a21c4131b38b0d753905206389`；40/40 个 seed-rate
> validation 记录均已完成并通过独立审计。

## 研究问题

本实验不训练新模型，而是冻结已有 Slot Missing-M3 checkpoint，判断 CMU-MOSI 的性能瓶颈更接近哪一种解释：

1. Missing-Latent Predictor 没有恢复足够的样本级缺失模态信息；
2. 已训练的 `missing_latent_fusion` 与 Emotion Head 无法利用目标 latent。

核心判别不是「真实 teacher latent 是否优于当前 predictor」，而是：在完全相同的 incomplete graph hidden、缺失位置与分类路径下，正确配对的真实 teacher latent 是否稳定优于同目标模态内打乱配对的 teacher latent。

该实验是使用 privileged information 的机制诊断，不是可部署方法，也不把 oracle 数值作为正式模型成绩。

## 锁定协议

- 数据集：CMU-MOSI official split；
- split：只迭代和评估 validation loader；公共数据对象会初始化三个 split，但
  test loader 不被迭代，test 样本不参与指标或任何实验决策；
- fold：1；
- seeds：66、67、68、69、70；
- missing rates：0.0、0.1、0.2、0.3、0.4、0.5、0.6、0.7；
- 模型：`classification_completion=true` 的 Slot Missing-M3；
- 历史模型代码基线：`c4874e4f8782044ebf283cf57f5b5a4e1bacb524`；
- task：MOSI regression，`evaluation_protocol=official`；
- graph：hidden 200、window 2/2、time attention false；
- MMoE：`dual-gate`；
- local context residual：false；
- mask：使用 checkpoint seed 与原 `ConversationMaskSchedule` 重建；
- 权重：全部冻结，不创建 optimizer，不调用 backward，不更新 EMA；
- checkpoint 选择：完全继承历史训练记录，不重新选择 epoch；
- 正式推理环境：`/data2/yb/reproduction_envs/gcnet-official/bin/python3.8`；
- 测试环境：`/data2/yb/reproduction_envs/s0/bin/python`；
- GPU：不得使用 GPU 4。

## 数据与 checkpoint provenance

### 特征

| 模态 | 特征目录名 |
|---|---|
| Audio | `wav2vec-large-c-UTT` |
| Text | `deberta-large-4-UTT` |
| Visual | `manet_UTT` |

正式运行使用以下数据根：

```text
/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features
```

标签文件由 runner 固定解析为特征目录上一级的：

```text
CMUMOSI_features_raw_2way.pkl
```

### 模型与历史记录

```text
checkpoint root:
/data2/yb/remote_experiments/missing_m3_mosi_classification_completion_20260829/formal

history root:
/data2/yb/paper/GCNet_TPAMI/.worktrees/missing-m3-sdr-backbone/experiments/missing_m3_mosi_classification_completion_20260829/results/formal
```

每个 seed 必须同时存在：

- `seed_<seed>/best.pt`；
- `seed_<seed>/history.json`；
- `seed_<seed>/config.json`。

Runner 会验证 checkpoint 的 key 集、配置、seed、保存 epoch 和 `validation_mean_weighted_f1` 均与历史记录一致。每个 seed/rate 还会记录 checkpoint SHA256、model state SHA256、历史 time-major mask SHA256、对齐后的 availability SHA256、target mask SHA256 和 sample-order SHA256。

## Temporal relation mapping 的历史缺陷

历史版 `batch_graphify` 通过 Python `set` 构造 temporal relation ID，导致 `past`、`now`、`future` 对应的 relation row 顺序可能随进程哈希状态改变。Checkpoint 保存了 RGCN relation weights，却没有保存当次训练所用的 relation-name-to-row mapping。因此，相同 checkpoint 在新进程中可能无法精确复现历史 validation 指标。

本诊断只用历史 validation 证据恢复该语义映射：

1. 枚举 `past`、`now`、`future` 的 6 种排列；
2. 对优先校准 rates 0.0、0.4、0.7 重算原生 predicted validation 指标；
3. 同时比较 W-F1、Macro-F1、Acc-2、MAE 和 Pearson correlation；
4. 仅接受五项指标最大绝对误差 `< 1e-6` 的候选；
5. 必须恰好恢复出 1 个候选，否则立即失败；
6. 对该 seed 的全部 8 个 rates 固定使用恢复后的映射。

该过程只重标 temporal edge type，不改变 edge topology、relation weight 或任何模型参数。恢复过程及全部 6 个候选误差都会写入结果 JSON。它不迭代 test loader，也不按 oracle 路径成绩选择映射，因此不是面向诊断结论的调参。

## 四条冻结推理路径

每个 seed/rate 只收集一次 observed encoder、GCNet graph hidden、原 Missing-Latent Predictor 和 EMA teacher 输出。有效 utterance 按 conversation-major 的 MOSI 指标顺序展平，四条路径共用同一 checkpoint、同一 incomplete graph hidden、同一 target mask、同一 classifier 和同一样本顺序。

| 路径 | 操作 | 作用 |
|---|---|---|
| `graph_only` | 直接将 `graph_hidden` 输入 `smax_fc` | 测量完全绕过 latent fusion 的分类基线 |
| `predicted` | 将原 predictor latent 经原 fusion 加回 `graph_hidden` | 精确复现已训练 completion 路径 |
| `real_teacher` | 将当前样本真实缺失模态的 EMA teacher latent 经同一 fusion 加回 | Privileged oracle 路径 |
| `shuffled_teacher` | 在相同 target modality 的缺失位置池内打乱 teacher latent 后，经同一 fusion 加回 | 控制模态身份、数值分布和 fusion 输入，只破坏样本配对 |

`graph_only` 必须直接绕过 fusion，不能把 latent 置零后送入 fusion，因为 fusion 的 Linear bias 可能产生非零 residual。

`predicted` 的手工路径必须同时在 hidden、logit 和五项 validation 指标上复现原生 model forward；最大绝对误差阈值为 `1e-6`。

## Shuffled teacher 负对照

- Audio、Text、Visual 分别置换，不跨 target modality；
- 只置换 `target_mask == 1` 的真实缺失位置；
- 不触碰 observed 位置和 padding；
- 使用稳定随机排序与非零循环位移；
- 当目标池大小至少为 2 时，不允许任何样本仍配对自身；
- seed 由 checkpoint seed、rate、shuffle index 与 target modality 稳定派生，不使用 Python `hash()`；
- 每个 seed/rate 先运行 8 个 shuffle；
- 若 shuffled W-F1 的 Monte Carlo 标准误差大于 0.001，自动扩展到 32 个 shuffle。

结果保存每次 shuffle 的指标、均值、标准差、有效次数、W-F1 Monte Carlo 标准误差，以及 target-wise permutation SHA256。

## 指标与统计

### 基础指标

四条路径均报告：

- Weighted F1；
- Macro F1；
- Accuracy（Acc-2）；
- MAE；
- Pearson correlation。

### 主比较

主比较为：

```text
real_teacher - mean(shuffled_teacher)
```

它检验正确的 utterance-level teacher 配对是否比仅保留目标模态身份和 latent 分布的负对照提供更多有效信息。

辅助比较为：

```text
real_teacher - predicted
```

该差值同时受到 predictor 误差与 teacher-to-fusion 分布外输入影响，不能单独作为 Predictor 能力的无偏估计。

### 重采样与跨 rate 汇总

- 每个 seed/rate 对 paired W-F1 delta 做 conversation-cluster bootstrap，同一 conversation 内 utterance 不拆分；
- 每条记录分别报告 `real_teacher - predicted` 与 `real_teacher - shuffled_teacher` 的 95% 区间；
- 非零 rate 汇总：0.1--0.7；
- 高缺失汇总：0.5--0.7；
- 先在每个 seed 内对指定 rates 求平均，再报告 5 seeds 的均值与标准差；
- 根目录汇总当前不计算跨 seed 的 cluster-bootstrap 区间，因此不得把「五种子均值为正」误写成「总体 95% 区间排除 0」。

只有五种子方向稳定，且对应的 per-record conversation-cluster bootstrap 证据一致时，才可称为稳定样本级增量。方向一致但区间包含 0 时只能称为弱证据。

## 表示与 fusion 诊断

为区分「teacher 本身无信息」与「teacher 对已训练 fusion 属于 OOD 输入」，每个 seed/rate 还记录：

- predicted/teacher target-wise raw channel std 与 centered effective rank；
- target-specific LayerNorm 后的 std 与 effective rank；
- fusion Linear 前后 RMS；
- `tanh` 饱和比例；
- fusion residual 范数；
- 相对 graph logit 改变量。

这些量只用于解释四路结果，不参与路径选择或参数更新。

## 完整性检查

正式结果必须同时满足：

- 40/40 seed-rate JSON 与 40/40 NPZ 存在；
- `predicted` 原生/手工 hidden 与 logit 最大误差 `< 1e-6`；
- 历史 validation 五项指标重算误差 `< 1e-6`；
- 40/40 运行前后 model state SHA256 相同；
- 40/40 named buffers 完整恢复；
- 所有输出均 finite；
- rate 0.0 的 target count 为 0，四路 prediction 最大误差 `< 1e-6`；
- NPZ 的 labels、availability、target mask 与四路 prediction 可独立重算 JSON 指标；
- `test_split_evaluated=false`。

## 正式运行命令

```bash
PYTHONHASHSEED=0 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
CUDA_VISIBLE_DEVICES=0 \
/data2/yb/reproduction_envs/gcnet-official/bin/python3.8 \
  scripts/run_mosi_latent_oracle.py \
  --feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features \
  --checkpoint-root /data2/yb/remote_experiments/missing_m3_mosi_classification_completion_20260829/formal \
  --history-root /data2/yb/paper/GCNet_TPAMI/.worktrees/missing-m3-sdr-backbone/experiments/missing_m3_mosi_classification_completion_20260829/results/formal \
  --output-dir /data2/yb/remote_experiments/missing_m3_mosi_latent_oracle_20260903/formal \
  --seeds 66 67 68 69 70 \
  --rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --shuffle-count 8 \
  --split validation \
  --device cuda \
  --code-commit 34bac0e69c4f13a21c4131b38b0d753905206389
```

## 结果

### 五种子逐 rate 汇总

表中数值为百分数；四路指标和 delta 均报告五种子均值 ± population SD
（`ddof=0`），delta 单位为百分点。`+seeds` 是该 rate 上配对 delta 为正
的 checkpoint 数；单个 rate 的 5 个 checkpoint 可能因 MCSE 规则使用不同 K。

| Miss | Graph-only W-F1 | Predicted W-F1 | Real teacher W-F1 | Shuffled teacher W-F1 | Real - Predicted | RP +seeds | Real - Shuffled | RS +seeds | Shuffle K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.89 ± 0.70 | 85.89 ± 0.70 | 85.89 ± 0.70 | 85.89 ± 0.70 | +0.000 ± 0.000 | 0/5 | +0.000 ± 0.000 | 0/5 | 8 |
| 0.1 | 83.15 ± 1.28 | 83.77 ± 0.98 | 82.97 ± 1.22 | 83.06 ± 1.32 | -0.798 ± 0.534 | 1/5 | -0.085 ± 0.203 | 3/5 | 8--32 |
| 0.2 | 81.92 ± 1.60 | 82.17 ± 1.41 | 82.11 ± 1.56 | 82.13 ± 1.61 | -0.053 ± 1.010 | 2/5 | -0.016 ± 0.308 | 2/5 | 8--32 |
| 0.3 | 78.94 ± 0.81 | 78.67 ± 1.87 | 79.04 ± 1.34 | 79.03 ± 1.35 | +0.368 ± 1.290 | 3/5 | +0.008 ± 0.302 | 2/5 | 8--32 |
| 0.4 | 76.46 ± 3.16 | 76.92 ± 2.97 | 76.88 ± 3.23 | 76.82 ± 3.40 | -0.044 ± 1.913 | 2/5 | +0.057 ± 0.533 | 2/5 | 8--32 |
| 0.5 | 73.55 ± 3.79 | 72.79 ± 2.54 | 73.31 ± 3.37 | 73.38 ± 3.39 | +0.513 ± 1.647 | 3/5 | -0.079 ± 0.238 | 2/5 | 8--32 |
| 0.6 | 73.35 ± 2.48 | 73.92 ± 1.73 | 73.11 ± 2.57 | 73.07 ± 2.32 | -0.813 ± 1.150 | 2/5 | +0.041 ± 0.417 | 2/5 | 32 |
| 0.7 | 73.94 ± 2.11 | 73.44 ± 1.99 | 74.22 ± 2.05 | 73.96 ± 2.23 | +0.781 ± 1.634 | 4/5 | +0.254 ± 0.585 | 4/5 | 32 |

### 预注册汇总

| Rate group | Real - Predicted | RP +seeds | Real - Shuffled | RS +seeds | 结论等级 |
|---|---:|---:|---:|---:|---|
| Nonzero（0.1--0.7） | -0.007 ± 0.358 | 3/5 | +0.026 ± 0.236 | 3/5 | 不支持稳定增益 |
| High missing（0.5--0.7） | +0.160 ± 0.466 | 3/5 | +0.072 ± 0.359 | 4/5 | 弱正点估计，未过稳定性门槛 |

汇总遵循预注册顺序：先在每个 seed 内对 rates 求均值，再跨 5 seeds
统计。非零 rate 的 35 个 `real - shuffled` 单元为 17 正、17 负、1 个
恰为零；其中 conversation-cluster bootstrap 只有 4 个显著正、3 个显著
负、28 个跨零。最有利的 rate 0.7 虽然有 4/5 seeds 为正，但也同时出现
显著正的 seed 66 和显著负的 seed 69，因此不能称为可重复增益。

### Provenance 与完整性结果

| 检查项 | 结果 |
|---|---|
| 结果网格 | 5 seeds × 8 rates = 40/40 |
| JSON / NPZ 数量 | 40 / 40，另有根 `summary.json` |
| 每条记录样本数 | 229 utterances / 10 conversations |
| 历史指标最大复现误差 | `1.356e-7` |
| 原生/手工 hidden 最大误差 | `7.153e-7` |
| 原生/手工 logit 最大误差 | `4.768e-7` |
| 原生/手工五指标最大误差 | `1.082e-8` |
| rate 0 四路最大误差 | `0.0` |
| state SHA 不变 | 40/40 |
| buffers 恢复 | 40/40 |
| finite 检查 | 40/40 |
| Shuffle 扩展 | 16 条 K=8；24 条 K=32 |
| test loader 是否迭代 / 评估 | 否 / 否；`test_split_evaluated=false` |

24 条记录按规则扩展到 K=32；seed 66/rate 0.7 和 seed 67/rate 0.4
达到上限后 MCSE 分别为 0.001011 和 0.001069，仍略高于阈值 0.001。
主效应量与这一误差处于同一量级，因此不能解释成微弱确定提升。

### 恢复出的 temporal relation order

| Seed | Relation row order |
|---:|---|
| 66 | `future, now, past` |
| 67 | `now, past, future` |
| 68 | `future, now, past` |
| 69 | `now, future, past` |
| 70 | `future, past, now` |

五个 checkpoint 均只存在一个 `<1e-6` 的候选。不同 seed 的语义排列确实
不同，证明该映射不能由新进程默认顺序或一个全局硬编码恢复；未来 checkpoint
必须显式保存 relation-name-to-row mapping。

## 诊断结果

### Completion 路径本身没有稳定收益

把当前 Predictor 路径与完全绕过 latent fusion 的 `graph_only` 比较：

- 非零 rates：Graph-only 为 77.330%，Predicted 为 77.382%，仅
  `+0.053 ± 0.330` 个百分点，seed 内平均 3/5 为正；
- 高缺失 rates 0.5--0.7：Graph-only 为 73.612%，Predicted 为
  73.383%，反而为 `-0.229 ± 0.476` 个百分点，只有 1/5 seed 为正。

因此，当前 completion residual 并未稳定改善 MOSI validation 分类；问题不能
简化成「把 Predictor 的 latent loss 再降一点即可」。

### Teacher 与 fusion 存在明显的几何错位

这里的 raw latent 是 Predictor/Teacher 送入 fusion 的表示，不是上游原始特征。
Teacher projector 自带 output LayerNorm，所以其 raw RMS 约为 1 是结构预期。
下表在全部非零 rates 上按记录取均值，格式为 `Predicted / Real teacher`：

| Target | Raw RMS | Raw effective rank | Fusion-LN RMS | Fusion-LN effective rank | Linear output RMS |
|---|---:|---:|---:|---:|---:|
| Audio | 0.530 / 1.000 | 15.14 / 45.66 | 0.987 / 0.986 | 17.74 / 45.66 | 0.133 / 0.054 |
| Text | 0.418 / 1.000 | 17.04 / 55.10 | 0.986 / 0.985 | 19.34 / 55.10 | 0.123 / 0.058 |
| Visual | 0.484 / 0.999 | 14.73 / 30.91 | 0.988 / 0.987 | 18.01 / 30.91 | 0.140 / 0.046 |

Fusion 的 LayerNorm 将两种输入的 RMS 都校准到约 0.986，但没有消除协方差方向
和 effective-rank 差异。只在 predicted manifold 上训练的 target-specific Linear
对 teacher 的响应明显更弱。全部非零 rates 汇总后：

| 量 | Predicted | Real teacher | Teacher / Predicted |
|---|---:|---:|---:|
| Fusion residual RMS | 0.1026 | 0.0419 | 0.408 |
| Fusion residual mean L2 | 1.8205 | 0.7482 | 0.411 |
| Logit-shift RMS | 0.3152 | 0.0762 | 0.242 |
| Logit-shift mean absolute | 0.2280 | 0.0536 | 0.235 |
| `tanh` saturation | 0 | 0 | -- |

因此 `real_teacher ≈ shuffled_teacher` 的一个直接混杂是：两者经过当前 fusion
后都只造成很弱的干预。该现象不是 raw norm 未归一化，也不是 `tanh` 饱和；
更符合 fusion Linear 已适配低秩 predicted 子空间、却对 teacher 的高秩方向低响应。

### 最终结论

在五个 checkpoint 和八个 missing rates 上，正确配对的 EMA teacher latent 相对
同目标模态内 shuffled teacher 没有产生稳定的 validation W-F1 增益。非零 rates
的平均差值只有 `+0.026` 个百分点，五个 seed 中三个为正；高缺失率为
`+0.072` 个百分点，但 seed 间方向和幅度明显不一致。

本实验因此排除了一个强假设：**把当前 prediction 直接替换成真实 EMA teacher
latent，再送入现有 fusion，并不会稳定大幅提分。** 它没有证明 Predictor 缺少
样本级信息，也没有证明 teacher latent 不包含情绪信息。当前阴性结果仍同时兼容
fusion-interface mismatch、不同目标模态贡献相互抵消，以及 teacher target 的
情绪相关性不足。

## 判读边界

- 若 `real_teacher` 稳定优于 `shuffled_teacher`，只支持「真实缺失模态中存在当前 predicted residual 未提供的、与样本配对有关的有效信息」；
- 上述阳性结果不能证明该信息必然能从 observed sources 恢复，也不能把差距单独归因于 Predictor 架构；
- 若 `real_teacher` 同时优于 `predicted`，仍需结合 OOD 统计判断 teacher 直接进入 fusion 是否改变了输入尺度、秩或饱和状态；
- 若 `real_teacher` 不优于 `shuffled_teacher`，只能说明在当前已训练 fusion/classifier 的直接注入接口下未观察到稳定样本级增量；不能证明 teacher latent 没有情感信息；
- `real_teacher` 的 graph hidden 仍来自 incomplete input，因此它不是完整模态性能上界；
- rate 0.0 没有真实缺失 target，只是四路等价性负控，不计入主效应；
- 本轮不迭代或评估 test loader，不能把 validation 结果作为最终泛化结论。

## 限制与后续边界

1. Fusion 与 classifier 只在 predicted latent 上训练，real teacher latent 对它们可能属于分布外输入；
2. 直接注入把「teacher 信息量」与「现有 fusion 可利用性」绑定在一起，无法彻底拆分；
3. Shuffled teacher 保留目标模态与总体分布，但不能控制所有局部语义或 conversation-level 结构；
4. 每个 seed/rate 的 bootstrap 以有限 validation conversations 为单位，区间可能较宽；
5. 当前根汇总只有跨 seed 均值与标准差，没有预注册的跨 seed hierarchical bootstrap；
6. 本轮不训练 supervised probe，不搜索新 fusion，也不修改模型；
7. 若直接注入结果为阴性，冻结 supervised probe 应作为独立后续诊断，不能回填或改写本轮结论；
8. 只有 validation 主比较呈现稳定样本级增量后，才讨论一次锁定协议的 test 验证。

另外，对 A/T/V singleton pattern，一次推理会同时替换两个缺失 target，再按原
fusion 规则汇总三个 target-specific residual。rate 0.7 中有 872/1145（76.2%）
utterances 属于双 target 注入，rate 0.4 为 423/1145（36.9%）。这对 Real、
Predicted 和 Shuffle 是公平的部署路径比较，但无法排除某个 target 单独有效、
联合时却被平均稀释或与另一 target 抵消。

本轮 checkpoint 又是按同一 validation split 上的 Predicted 路径成绩选择的，
因此 `real - predicted` 存在选择偏差，只能作为辅助诊断；主比较
`real - shuffled` 未参与 checkpoint 选择，但仍属于 selected-validation 机制筛选。

最小后续诊断不是重新训练主模型，而是冻结 GCNet、Predictor 和 Teacher，分别对
Audio/Text/Visual 使用相同容量的闭式 ridge residual probe，仅在 train 上拟合、
在 validation 上比较 paired teacher、predicted 与 target-wise shuffled teacher。
它可以隔离当前 fusion 的子空间错位；该后续实验必须另立协议和结果目录，不能
回填本轮预注册结论。

## 结果文件

已生成目录结构：

```text
results/
├── summary.json
├── seed_66/
│   ├── rate_0p0.json
│   ├── rate_0p0.npz
│   └── ...
├── seed_67/
├── seed_68/
├── seed_69/
└── seed_70/
```

每个 NPZ 保存 sample keys、conversation IDs、utterance indices、labels、availability、target mask、graph-only/predicted/real-teacher predictions 和全部 shuffled-teacher predictions，供指标独立复算。
