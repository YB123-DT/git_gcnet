# 40 个 OSRAM 候选：代码、迁移思路与收益假设

**INTERNAL DIAGNOSTIC ONLY**

核对日期：2026-10-04。本文整理两轮已经接纳的 20 + 20 项，不新增候选，不修改训练代码，不重新运行实验。整理基于工作树 HEAD `b227161cf87bd3efa53572e2f74f313c6c05150e`；这不是历史 Flat baseline 的训练 commit。

## 先回答：这 40 个到底是什么

它们是 **40 个借鉴其他领域论文机制、接入 OSRAM 的模块变体**，不是 40 个原论文完整模型复现。原论文有依据，不等于迁移到 MOSI 就有收益；代码存在、接口检查通过，也不等于任务效果成立。

- [第一轮 20 项逐项清单](osram40_round1_inventory.md)：实际类和代码位置、移植机制、加入理由、收益假设、迁移边界与最终 seed66 分数。
- [第二轮 20 项逐项清单](osram40_round2_inventory.md)：实际类和代码位置、哪些输入被改动、加入理由、收益假设及迁移边界。
- 原始来源与设计记录：[ROUND1.json](../experiments/osram_meaningful20_20261003/ROUND1.json)、[ROUND2.json](../experiments/osram_meaningful20_round2_20261004/ROUND2.json) 及后者引用的各候选卡。

## 两轮接入位置不同

记 `L` 为 Local，`C=[B,G_A,G_T,G_V]` 为已读出的有效历史证据，`S` 为原 Local Skip，`A` 为原 emotion_adapter。

原 Flat：

```text
h = emotion_norm(S(L) + A([L,C]))
```

第一轮：保留原 Flat anchor，在 norm 前加模块残差。

```text
h = emotion_norm(S(L) + A([L,C]) + W(core(L,C)))
```

入口：[MeaningfulReadoutResidual](../gcnet_missing_m3/meaningful_blocks.py#L28)；接回位置：[osram.py](../gcnet_missing_m3/osram.py#L1733)。`W` 零初始化。多数 core 输出 128 维，但 TabNet 输出 64、NODE 输出 192，包装层按 `core.output_dim` 投影到真实 output_dim，而非假定全部 128。

第二轮：处理送入原 Flat adapter 的输入，**不改变 Local Skip 的输入**。

```text
(L_new,C_new) = core(L,C)
h = emotion_norm(S(L) + A([L_new,C_new]))
```

入口：[MeaningfulInputAdapter](../gcnet_missing_m3/meaningful_input.py#L33)；接回位置：[osram.py](../gcnet_missing_m3/osram.py#L1715)。各 core 是否改 Local 见逐项清单，不能把这一轮统称为“只过滤 Memory”。

共同边界：

- 使用已经读出的 forward 512 = 8×64 历史维度；不是对 backward 恒零半区学习新历史。
- 保留固定 Base/Gap 槽位、availability 与 padding 屏蔽；首句跳过新历史处理。
- 不改变 OSRAM 因果 Memory 读写/query、ObservedSetEncoder、原任务头及任务目标；不加入 JEPA、补全、双视图或新的辅助 loss。
- 新模块内部的 token、graph、slot、MAC reasoning state 等，不是第二条跨 utterance 的 OSRAM Memory 轨迹。
- 初始零残差或 identity bridge 只保证起点接近/等价原路径，不保证联合训练后主干不漂移，也不保证提分。

## 为什么当时加入：五类假设，而不是五个已经证实的瓶颈

| 假设方向 | 对应候选 | 可能的收益（尚未证明） | 不能由此声称 |
|---|---|---|---|
| 显式组织多路证据关系 | RRN、EGT、GatedGCN、PNA、AllSet、ED-HNN、Hyper-SAGNN、Sheaf、DGCNN、PPGN | 把 Local、Base、Gap 或各 head 之间的组合关系显式编码，让 Flat 不必单独从拼接中学出全部关系 | 已检测到真实冲突或 emotion shift |
| 聚类、分组、压缩后再使用 | Capsules、Slot Attention、OTKE、Perceiver、GMT、TokenLearner、MBT、NetVLAD、ToMe | 组织重复/互补证据，给读出提供更紧凑的中间结构 | 压缩必然去噪，或这里必然更快 |
| 条件检索与多步交互 | Differential、Flowformer、Compositional、Dual Attention、DCFormer、MCAN、Dense Co-Attention、MAC | 根据当前信息改变历史证据的组合方式，比固定拼接提供更有结构的条件交互 | 学到了缺失 Text，或历史有害时必然会关闭它 |
| 分解、迭代求解和几何表示 | Hamburger、CRATE、Equilibrium Aggregation、Spline、Learned RPCA、DEQ、SPDNet | 提供低秩/稀疏、迭代平衡、非线性校准或二阶统计等归纳偏置 | 已证明输入噪声满足分解假设，或有限展开等于严格最优解 |
| 特征选择与结构化集合推理 | TabNet、NODE、NLM、FSPool、RAT-SPN | 表达选择、条件分支、集合排序或结构化组合，补充普通 MLP 的函数形式 | 获得可解释因果规则，或归一化概率模型就是任务校准概率 |

这里一些方法横跨多个方向。它们的机制不同，但不少仍检验相近的“让读出更有结构”假设，不能当作 40 个完全独立的科学假设。来源来自非 MSA/MERC 领域，也不自动构成我们任务的创新或必要性。

## 当前效果证据

统一结果口径：MOSI seed66，100 epochs，8 rates 等权平均；high 为 .5/.6/.7 等权平均；**per-rate TEST-oracle**，不是 validation 选模的正式论文成绩。多候选反复筛选进一步增加测试集选择偏差。

| 状态/模型 | 8-rate W-F1 (%) | High W-F1 (%) | 相对 Flat 的 8-rate 差值（百分点） |
|---|---:|---:|---:|
| 对应原 Flat seed66 | 81.068 | 76.352 | — |
| 第一轮最好：Perceiver IO | 80.818 | 76.126 | −0.250 |
| 第二轮完成：Dense Co-Attention | 80.286 | 75.566 | −0.782 |
| 第二轮完成：MAC | 80.171 | 75.383 | −0.897 |
| 第二轮完成：SPDNet | 80.024 | 75.227 | −1.044 |
| 第二轮完成：MCAN | 79.960 | 75.272 | −1.108 |

来源：[第一轮 SUMMARY.json](../experiments/osram_meaningful20_20261003/SUMMARY.json)、[第二轮已完成结果](../experiments/osram_meaningful20_round2_20261004/PARTIAL_RESULT.md)。第一轮 20 项全部完成，全部低于 Flat；第二轮完成的四项也都低于 Flat。**已完成 24 项没有显示 overall 超越基线的收益。** 不报告单 seed 显著性，不用部分 epoch 的临时最好分数代替完整结果。

远程 `biggpu` 的 `QUEUE.json` 与对应 `metrics.json` 存在性于 **2026-10-04 14:10:48 UTC** 只读复核：第二轮 4 complete / 12 running / 4 尚未启动。该时间点状态不是永久状态，未完成不等于失败。

正在运行：Differential、Flowformer、Compositional、Dual Attention、DCFormer、Spline、Learned RPCA、DEQ、TokenLearner、NLM、FSPool、RAT-SPN。待运行：MBT、NetVLAD、ToMe、PPGN。本次整理未改变任何运行中的任务。

## 这份整理应该怎样使用

1. **看实现，不只看论文名字。** 逐项清单区分原论文目标和当前固定维度、固定证据槽位下的迁移版本。
2. **看结果，不把故事当结论。** 当前最强证据是“这些已完成的迁移没有在固定协议下超过 Flat”，不是“这些论文无用”，更不是“已定位根因”。
3. **两轮不能直接解释为机制胜负。** 插入位置、参数量与内部优化形式并未逐一匹配；没有同容量对照时不能把差异归因于特定结构本身。
4. **已有证据的上限没被突破。** 这些模块处理当前 Local 和已读历史，不能创造当前与历史均未提供的模态信息。
5. **保留边界与失败记录。** 不把近似实现写成原论文完整复现，不把有代码写成理论保证，不因名字复杂就自动进入更多种子。

本轮静态核对也保留一个实现限制：NLM 总计 198,644 参数中，18,064 个参数不连到最终 task 读出（末层二元/三元推断与倒数第二层三元推断）。这意味着总参数量不等于全部有效训练容量。该限制应随候选一起记录；本次整理不修改正在运行的版本。

历史 baseline 训练 commit 尚未在已有审计中得到确认；baseline 指标和文件指纹见既有 [BASELINE_AUDIT](../experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json)。第二轮训练 source 为 `98268f4d179fc274c3b4487cf698faec2e293a8f`，不要用本文整理 commit 替代真实训练版本。
