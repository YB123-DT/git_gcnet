# IEMOCAP-6 Equal-Budget Stratified Missing-M3 JEPA Ablation

## 研究问题

在不把每个 source batch 复制成 8 个 missing-rate views 的前提下，Missing-M3 的 JEPA 梯度是否仍能稳定改善 IEMOCAP-6？

本实验只比较两个参数、mask、batch order、graph 和训练配置均配对的 arms：

- **With-JEPA**：`train_rate_mode=stratified`，`jepa_weight=0.1`；
- **JEPA-gradient-off**：`train_rate_mode=stratified`，`jepa_weight=0.0`。

它不是与 Original GCNet 的最终比较。`gradient-off` 只隔离 JEPA loss 的贡献；通过本轮后才能运行 matched stratified Original control。

## 锁定协议

- Dataset：IEMOCAPSix，fold 5；
- Seeds：66、67、68、69、70；
- Missing rates：0.0–0.7；
- 训练：100 epochs，batch size 32；
- 每段训练对话每 epoch 只生成一个 masked view；
- 每 epoch 120 段对话、120 个 masked views、4 次 model forward、4 次 optimizer step；
- 每个请求 rate 每 epoch 恰好分配 15 段对话；
- checkpoint 由八个 rate 的 validation W-F1 均值选择；
- 同一个选中 checkpoint 测试全部八个 rate；
- Source commit：`6c5346b8cad453920689b2806562eb578f47b424`。

## 完整性与配对审计

审计结果为 **PASS**：

- 10/10 jobs 完成，runner failures 为 0；
- 10 个 history 均为 100 epochs，共核验 1,000 个 epoch records；
- 80/80 prediction NPZ 存在，并独立复算 W-F1、Macro-F1、Accuracy，与 `metrics.json` 一致；
- 两臂 500/500 个 epoch assignment hash 相同；
- 40/40 个 seed-rate test mask hash 相同；
- 40/40 个 labels 与 availability 数组逐元素相同；
- 每个 epoch 的 raw missing/total counts 可精确复算 realized missing fraction；
- 所有 80 个预测均覆盖 6 个类别，无单类坍塌。

跨五种子、100 epochs 的每臂 sampling 总账为：

| η | Conversations | Valid utterances | Missing / Total modality elements | Realized η | JEPA targets |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 7,500 | 362,354 | 0 / 1,087,062 | 0.000000 | 0 |
| 0.1 | 7,500 | 364,312 | 108,557 / 1,092,936 | 0.099326 | 108,557 |
| 0.2 | 7,500 | 364,245 | 215,783 / 1,092,735 | 0.197471 | 215,783 |
| 0.3 | 7,500 | 362,434 | 316,564 / 1,087,302 | 0.291146 | 316,564 |
| 0.4 | 7,500 | 365,070 | 415,309 / 1,095,210 | 0.379205 | 415,309 |
| 0.5 | 7,500 | 363,206 | 498,505 / 1,089,618 | 0.457504 | 498,505 |
| 0.6 | 7,500 | 362,680 | 575,116 / 1,088,040 | 0.528580 | 575,116 |
| 0.7 | 7,500 | 360,699 | 634,190 / 1,082,097 | 0.586075 | 634,190 |

高 missing rate 的 realized η 低于请求值，是每个有效 utterance 至少保留一个模态的 nonempty repair 所导致的预期结果。

轻量结果均保存在 [results](results/)；`best.pt` 未同步进仓库。

## Test W-F1（%）

### With-JEPA

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 八率均值 | 0.4–0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 67.97 | 66.43 | 67.16 | 66.96 | 67.85 | 65.86 | 64.55 | 63.05 | 66.23 | 65.33 |
| 67 | 64.80 | 65.42 | 64.26 | 64.63 | 63.08 | 62.66 | 60.47 | 60.35 | 63.21 | 61.64 |
| 68 | 65.08 | 66.42 | 65.42 | 64.45 | 64.63 | 64.68 | 63.14 | 63.58 | 64.67 | 64.00 |
| 69 | 64.41 | 64.73 | 63.59 | 64.34 | 62.77 | 64.78 | 62.47 | 61.63 | 63.59 | 62.91 |
| 70 | 66.54 | 65.87 | 66.83 | 66.16 | 64.31 | 62.64 | 61.78 | 59.29 | 64.18 | 62.00 |
| **Mean** | **65.76** | **65.77** | **65.45** | **65.31** | **64.53** | **64.12** | **62.48** | **61.58** | **64.38** | **63.18** |

### JEPA-gradient-off

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 八率均值 | 0.4–0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 65.43 | 65.30 | 65.06 | 64.06 | 65.36 | 62.97 | 63.11 | 60.32 | 63.95 | 62.94 |
| 67 | 65.46 | 64.97 | 65.72 | 67.84 | 66.45 | 63.46 | 62.50 | 59.01 | 64.42 | 62.85 |
| 68 | 67.03 | 65.99 | 65.22 | 64.23 | 65.50 | 63.86 | 62.33 | 63.19 | 64.67 | 63.72 |
| 69 | 67.57 | 68.35 | 66.15 | 65.83 | 64.87 | 65.45 | 63.84 | 63.70 | 65.72 | 64.46 |
| 70 | 63.98 | 64.17 | 61.82 | 63.16 | 60.82 | 61.55 | 59.32 | 58.14 | 61.62 | 59.96 |
| **Mean** | **65.89** | **65.75** | **64.79** | **65.02** | **64.60** | **63.46** | **62.22** | **60.87** | **64.08** | **62.79** |

## Paired JEPA effect

| Rate | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean delta（百分点） | −0.13 | +0.02 | +0.66 | +0.28 | −0.07 | +0.67 | +0.26 | +0.71 |
| Positive seeds | 2/5 | 4/5 | 3/5 | 3/5 | 2/5 | 3/5 | 3/5 | 4/5 |

按 seed 聚合：

| Seed | 八率 mean delta（百分点） | 0.4–0.7 mean delta（百分点） |
| ---: | ---: | ---: |
| 66 | +2.28 | +2.39 |
| 67 | −1.22 | −1.21 |
| 68 | +0.01 | +0.28 |
| 69 | −2.13 | −1.55 |
| 70 | +2.56 | +2.04 |
| **Mean** | **+0.30** | **+0.39** |

预注册门槛结果：

- 八率总体 mean delta > 0：通过；
- 高缺失率 mean delta > 0：通过；
- 八率 seed-level delta 至少 3/5 为正：3/5，通过；
- 高缺失率 seed-level delta 至少 3/5 为正：3/5，通过。

因此本轮 directional gate 为 **PASS**。

## 结论边界

JEPA 在等训练 view/forward/update 预算下仍保留小幅正向信号，说明此前收益不完全由 `all-rates-per-batch` 的 8 倍 masked-view 预算制造。正向主要出现在 0.2、0.3、0.5、0.6、0.7；0.0 和 0.4 的五种子均值略降。

但效应还不能称为稳定显著提升：

- 八率 seed-level delta 的 95% t 区间为 −2.28 到 +2.88 个百分点；
- 高缺失率区间为 −1.85 到 +2.64 个百分点；
- 两个区间都跨 0；paired t-test 分别为 `p=0.764` 和 `p=0.654`；
- seed 66/70 的正增益抵消了 seed 67/69 的负增益，seed 68 的八率均值仅提高 0.006 个百分点。

另外，trainer 没有把 validation mask hash 写入 history。两臂的源码、seed、split、fold、sampler 和固定 schedule 配置完全相同，独立重建也一致，但这里没有将 validation hash 计入落盘 artifact-level 配对证据。Official IEMOCAP 的 validation/test 共享 held-out Session，因此绝对分数也不能表述为严格独立验证后的泛化结果；这不影响同协议 A/B 配对，但限制外部结论。

准确表述是：**通过预注册的最小方向性筛选，值得运行 matched stratified Original GCNet control；当前证据不足以声称稳定优于 Original 或统计显著。**

## Next action

按设计文档下一阶段运行同一 stratified sampler、同一训练预算的 Original GCNet control。只有在该控制下仍显示优势，才扩展到其他数据集；旧 `all` 结果继续只作为额外 view-budget 消融。
