# IEMOCAP-6 Matched Stratified Original GCNet Control

## 研究问题

在每段 source conversation 只产生一个 masked view，并严格匹配训练 rate、
batch order、mask、epoch、优化器和选模规则时，当前 Missing-M3 是否优于
Original GCNet？

本实验形成三个身份不可互换的 arms：

- **With-JEPA**：Missing-M3 架构，JEPA 权重为 0.1；
- **JEPA-gradient-off**：同一 Missing-M3 架构，JEPA 权重为 0；
- **Matched Original GCNet**：Original GCNet 架构及其分类加 masked
  reconstruction 目标，不含 Student、EMA teacher、MMoE 或 missing-latent
  predictor。

因此，`With-JEPA - gradient-off` 隔离 JEPA 梯度，`gradient-off - Original`
反映非 JEPA 整体差异（模型参数化以及 Original reconstruction 目标同时不同），
`With-JEPA - Original` 才是完整方法比较。

## 锁定协议

- Dataset：IEMOCAPSix，fold 5；
- Seeds：66、67、68、69、70；
- Requested missing rates：0.0--0.7；
- 100 epochs，batch size 32，Adam，学习率 `1e-3`，weight decay `1e-5`；
- hidden 200，window 2/2，dropout 0.5，time attention 关闭；
- 每 epoch 120 段 source conversations、120 个 masked views、4 次 forward、
  4 次 optimizer step；
- 每个 requested rate 每 epoch 恰好分配 15 段 conversation；
- checkpoint 由八率 validation W-F1 均值选择；
- 恢复唯一选中 checkpoint 后测试全部八个 rate。

Original 使用 corrected formal masked reconstruction loss，权重为 1.0，且
`reccls_flag=false`。它保留 Original 的训练目标，但共享相同的 stratified
预算；所以应称为 **matched-protocol Original control**，而不是 upstream
逐 rate 训练协议的原样复现。

## 完整性与配对审计

审计结果为 **PASS**：

- Original 5/5 jobs 完成，runner failures 为 0；三臂合计 15 jobs；
- 15 个 history 共 1,500 个 epoch records；
- 120/120 prediction NPZ 存在，独立复算 360 个 W-F1、Macro-F1、Accuracy，
  均与 `metrics.json` 一致；
- 三个 pairwise comparisons 各有 500/500 个 epoch assignment SHA、40/40 个
  test mask SHA、40/40 个 labels 和 40/40 个 availability 数组完全相同；
- Original 每个 epoch 均满足 120 source / 120 view / 4 forward / 4 update；
- Original reconstruction target 数严格等于自然缺失模态数；
- 三臂所有 120 个预测 bundle 都覆盖六个类别，无单类坍塌或 NaN/Inf；
- Original 参数量为 34,140,166，反而高于 Missing-M3 的总参数 32,212,238
  和可训练参数 31,352,078，因此 Missing-M3 的优势不能用参数更多解释。

Original 新增落盘了 validation mask SHA；既有两个 Missing-M3 arms 没有保存
该字段，所以三臂 validation 的 artifact-level hash 配对不计数。它们使用相同
split、seed 和 deterministic schedule，但不能把这一点写成已落盘的 hash 证据。

跨五种子、100 epochs 的公共 sampling 总账为：

| η | Conversations | Valid utterances | Missing / Total modality elements | Realized η |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 7,500 | 362,354 | 0 / 1,087,062 | 0.000000 |
| 0.1 | 7,500 | 364,312 | 108,557 / 1,092,936 | 0.099326 |
| 0.2 | 7,500 | 364,245 | 215,783 / 1,092,735 | 0.197471 |
| 0.3 | 7,500 | 362,434 | 316,564 / 1,087,302 | 0.291146 |
| 0.4 | 7,500 | 365,070 | 415,309 / 1,095,210 | 0.379205 |
| 0.5 | 7,500 | 363,206 | 498,505 / 1,089,618 | 0.457504 |
| 0.6 | 7,500 | 362,680 | 575,116 / 1,088,040 | 0.528580 |
| 0.7 | 7,500 | 360,699 | 634,190 / 1,082,097 | 0.586075 |

高 rate 的 realized missing fraction 低于 requested rate，是每个有效 utterance
至少保留一个模态的 nonempty repair 所致，三臂完全相同。

## Test W-F1（%）

### With-JEPA

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 八率均值 | 0.4--0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 67.97 | 66.43 | 67.16 | 66.96 | 67.85 | 65.86 | 64.55 | 63.05 | 66.23 | 65.33 |
| 67 | 64.80 | 65.42 | 64.26 | 64.63 | 63.08 | 62.66 | 60.47 | 60.35 | 63.21 | 61.64 |
| 68 | 65.08 | 66.42 | 65.42 | 64.45 | 64.63 | 64.68 | 63.14 | 63.58 | 64.67 | 64.00 |
| 69 | 64.41 | 64.73 | 63.59 | 64.34 | 62.77 | 64.78 | 62.47 | 61.63 | 63.59 | 62.91 |
| 70 | 66.54 | 65.87 | 66.83 | 66.16 | 64.31 | 62.64 | 61.78 | 59.29 | 64.18 | 62.00 |
| **Mean** | **65.76** | **65.77** | **65.45** | **65.31** | **64.53** | **64.12** | **62.48** | **61.58** | **64.38** | **63.18** |
| **SD** | 1.47 | 0.72 | 1.56 | 1.18 | 2.02 | 1.42 | 1.52 | 1.80 | | |

### JEPA-gradient-off

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 八率均值 | 0.4--0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 65.43 | 65.30 | 65.06 | 64.06 | 65.36 | 62.97 | 63.11 | 60.32 | 63.95 | 62.94 |
| 67 | 65.46 | 64.97 | 65.72 | 67.84 | 66.45 | 63.46 | 62.50 | 59.01 | 64.42 | 62.85 |
| 68 | 67.03 | 65.99 | 65.22 | 64.23 | 65.50 | 63.86 | 62.33 | 63.19 | 64.67 | 63.72 |
| 69 | 67.57 | 68.35 | 66.15 | 65.83 | 64.87 | 65.45 | 63.84 | 63.70 | 65.72 | 64.46 |
| 70 | 63.98 | 64.17 | 61.82 | 63.16 | 60.82 | 61.55 | 59.32 | 58.14 | 61.62 | 59.96 |
| **Mean** | **65.89** | **65.75** | **64.79** | **65.02** | **64.60** | **63.46** | **62.22** | **60.87** | **64.08** | **62.79** |
| **SD** | 1.43 | 1.59 | 1.72 | 1.85 | 2.19 | 1.42 | 1.73 | 2.48 | | |

### Matched Original GCNet

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 八率均值 | 0.4--0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 64.31 | 63.03 | 65.14 | 64.30 | 62.17 | 66.01 | 59.78 | 61.37 | 63.26 | 62.33 |
| 67 | 60.00 | 60.17 | 59.23 | 62.59 | 61.64 | 57.49 | 58.14 | 54.79 | 59.26 | 58.01 |
| 68 | 61.37 | 61.49 | 61.12 | 59.28 | 59.63 | 59.39 | 56.55 | 58.66 | 59.69 | 58.56 |
| 69 | 61.23 | 61.08 | 62.80 | 60.65 | 61.09 | 60.03 | 57.30 | 58.25 | 60.30 | 59.17 |
| 70 | 65.24 | 64.83 | 63.28 | 62.39 | 61.99 | 62.99 | 60.62 | 60.48 | 62.73 | 61.52 |
| **Mean** | **62.43** | **62.12** | **62.32** | **61.84** | **61.30** | **61.19** | **58.48** | **58.71** | **61.05** | **59.92** |
| **SD** | 2.23 | 1.83 | 2.24 | 1.93 | 1.02 | 3.34 | 1.70 | 2.54 | | |

## 三种 paired effects

| Comparison | 八率 mean delta | Positive seeds | 95% CI | p（双侧，未校正） | 高率 delta | Positive seeds | 95% CI | p（双侧，未校正） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| With-JEPA − gradient-off | +0.30 | 3/5 | [−2.28, +2.88] | 0.764 | +0.39 | 3/5 | [−1.85, +2.64] | 0.654 |
| With-JEPA − Original | **+3.33** | **5/5** | **[+1.71, +4.95]** | **0.0047** | **+3.26** | **5/5** | **[+1.02, +5.49]** | **0.0155** |
| gradient-off − Original | +3.03 | 4/5 | [−0.73, +6.79] | 0.089 | +2.87 | 4/5 | [−1.05, +6.78] | 0.112 |

`With-JEPA - Original` 的 rate-wise mean delta 从 η=0.0 到 0.7 分别为：

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Delta（百分点） | +3.33 | +3.66 | +3.14 | +3.47 | +3.22 | +2.94 | +4.00 | +2.87 |
| Positive seeds | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 3/5 | 5/5 | 4/5 |

这些 p 值是五个 paired seeds 上的描述性双侧 t-test，未做多重比较校正。
完整方法相对 Original 的两个聚合区间不跨零，但样本量仍只有五个 seeds，
不应把它扩大表述为跨数据集普适显著性。

## 结论

当前最可靠的结论是：

1. **完整 Missing-M3 在 matched equal-budget 协议下明显优于 Original GCNet。**
   八率均值从 61.05% 提升到 64.38%，高缺失率从 59.92% 提升到 63.18%；
   两个 aggregate comparison 均为 5/5 seeds 正向。
2. **大部分绝对差异在关闭 JEPA 梯度后已经存在。** gradient-off 相对
   Original 的均值为 +3.03 个百分点，但 seed 70 为负且置信区间跨零；JEPA
   在该架构内再增加 +0.30 个百分点，方向性门槛通过，但并不稳定显著。
   这 +3.03 同时包含模型参数化变化与 Original reconstruction 目标差异；没有
   classification-only Original 消融时，不能继续归因到某一个架构模块。
3. **合理的论文叙事不是“JEPA 单独带来 3.33 点”。** 更准确的表述是：
   非 JEPA 的 observed-set/slot 模型包已伴随主要差异，JEPA 提供较小的附加
   正则化收益，并让完整方法相对 Original 的 paired seed 结果达到 5/5 正向。

边界：Official IEMOCAP 的 validation 和 test 使用同一 held-out Session，所以上述
结果是严格同协议的内部 paired comparison，不能表述为具有独立 validation 的外部
泛化结论。其他数据集是否成立仍需单独验证。

原始轻量 artifacts、预测 NPZ、完整机器可读统计位于 [results](results/)；远程
checkpoint `best.pt` 未同步进仓库。
