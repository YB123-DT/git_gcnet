# Local / Base / Gap 读出侧实验去重回顾

本次只核对已有代码与结果；没有训练、新增推理或启动队列。
审查代码基点为 `9825582`，本报告仅增加整理文档。
范围为当前 cfg84 no-JEPA 的读出侧主要方法家族及相关训练对照，不宣称逐项验收
全部历史候选。模块目录数量、配置数量和已完成训练数量不能混为一谈。
旧目录中的 launch/running 段落可能是历史快照；优先使用完成报告和归档指标。

## 一、按实际职责归类，而不是按论文名计数

令 C=[Base,masked Gap-A/T/V]，原路径为
`head(LN(Skip(L)+Adapter([L,C])))`。

| 已试职责 | 代表实现 | 实际改动 | 去重结论 |
|---|---|---|---|
| 改历史总体强度 | History-input Gate、固定 history_scale 干预 | C'=alpha(L,C,a) C | 一个系数控制全部历史已经试过；固定缩放属于推理诊断 |
| 改各路证据强度 | Local-conditioned Evidence Gate、L1/L2、Gap-only、Gap-increment Filter | 各路标量门；部分版本只改Gap | 不能把“用Local决定Gap是否通过”当新职责 |
| 改证据内部通道及来源竞争 | Feature-only、Hierarchical Evidence Gate | 通道门；后者再接active-count归一化证据softmax | 层级版本是通道门，不应误写成已验证独立head路由 |
| 改Local使用方式 | Local Skip Gate、Memory-only Adapter | 调Skip；或Adapter去Local但保留Skip | 不是只控制过Memory；Local两条使用路径也试过 |
| 从当前—历史差异构造Flat残差 | Memory-Shift-Residual / D3-W256、Current–History Relation | 差分/乘积/绝对差→残差→原emotion_norm | “冲突/shift关系编码”已试，不是尚未覆盖的空白 |
| 改Flat输出表示 | Post-GRN；大量priority40候选 | 原Flat后门控残差，或pre-norm64→1600零初始化桥 | 不同算子不等于不同功能假说 |
| 改证据之间的交互后修正输入 | 原Nested、Neural Production、CWN等 | [L+deltaL,C+deltaC]进入保留的Flat | 不等于1600维Flat残差，也不等于替换Flat |
| 直接变换Flat输入 | Direct Nested/XCA/NPS/GATv2/PNA/CWN/Perceiver | Adapter接[L',C']，不在外部加原证据；Skip仍原L | “拿掉残差、让模块真正处理证据”已经训练过 |
| 改最终决策组织 | Evidence-centered Decision Correction | sL→sLB→sLBG，中心化增量，三个出口均监督 | 已替换原Flat决策路径；不是没试过logit correction |
| 改容量/训练责任分配 | 小Flat+Nested、图宽度、冻结Flat+Nested、双读出监督 | 缩Flat/扩大图/只训分支/监督原路径与完整路径 | “Flat太强压住模块”的若干可操作版本也已试过 |
| 拼接独立图特征 | Nested rooted concat+容量对照 | 原4352维输入追加Z320，保留Flat/Skip | 已实现并有启动证据；本地只有进度，不能当新提议或宣布最终失败 |

代码核对：`gcnet_missing_m3/osram.py` 的 memory-shift-residual、
meaningful_input_mode、nested_concat_mode 分支；`meaningful_input.py`；
`decision_correction.py`。ObservedSetEncoder/Local 与写入K/V本来就是不同投影路径，
不能把“第一次分开当前与存储表示”作为新贡献。

## 二、可比的100epoch三种子结果

以下均为MOSI seeds66/67/68、100epochs、八率W-F1均值，单位%。
**INTERNAL DIAGNOSTIC ONLY：历史 per-rate BEST Test-oracle，不是单checkpoint部署分数。**
不与本轮MC-FWC验证集选点结果混排。

| 方法 | 八率 | 高缺失 | 八率Δ vs Flat |
|---|---:|---:|---:|
| Flat |80.559|75.594|—|
| Nested原版 |80.489|75.252|−0.070|
| Neural Production |80.459|75.558|−0.100|
| Relation+双读出监督 |80.315|75.366|−0.244|
| Local Evidence Gate+L2 |80.286|75.955|−0.273|
| XCiT-XCA |80.213|75.365|−0.346|
| Evidence Gate+L1 |80.203|75.153|−0.356|
| Local Skip Gate |80.193|75.331|−0.366|
| Hierarchical Evidence Gate |80.018|75.267|−0.541|
| Feature-only Gate |79.957|75.158|−0.602|
| Memory Shift D3-W256 |79.953|74.992|−0.606|
| Memory Shift D1-W128 |79.949|75.206|−0.610|
| History-input Gate |79.892|75.239|−0.667|
| Memory-only Adapter |79.679|74.600|−0.880|
| Gap-increment Filter |79.579|74.699|−0.980|
| Post-GRN |79.521|74.665|−1.038|
| Decision Correction |79.290|74.164|−1.269|

来源：
- [top3三种子](../osram_readout_top3_3seed_20261005/RESULT.md)
- [Relation双读出三种子](../osram_relation_dual_readout_20261003/MULTISEED.md)
- [L2 Gate](../osram_local_evidence_gate_20260930/RESULT.md)
- [D3-W256三种子](../osram_shift_capacity_20261002/THREE_SEED_RESULT.md)
- [其他读出已完成汇总及各自路径](../osram_relation_dual_readout_20261003/NEXT_CANDIDATE.md)
- [Decision Correction三种子](../osram_decision_correction_20261003/RESULT.md)

整体均未超过Flat，不等于所有条件都无收益。L2 Gate高缺失+0.361pp，主要来自
rate.6；seed66高缺失仍略降，不能宣称各seed一致改善。Nested/NPS接近Flat，
但“最接近”不是已经提供增益。D3-W256单seed80.932不能代替三种子79.953。

## 三、直接替换也不是空白

MOSI seed66、100epochs、同类Test-oracle：Flat81.068/high76.352。

| 直接输入版本 | 八率 | 高缺失 |
|---|---:|---:|
| Nested随机初始化64/64 |80.177|74.800|
| Neural Production |79.971|75.377|
| PNA |79.685|74.618|
| GATv2 |78.759|73.263|
| Perceiver IO |78.362|72.926|
| CWN |78.316|72.777|
| XCA |75.898|71.261|

这些保留Flat，不是“删除Flat的七组”。Nested直接版本另有8个已完成节点/GIN宽度
配置，最佳80.122，未超过原直接64/64或Flat。部分适配同时改变解码接口、参数量和
初始化，因此不能从下降断言“残差普遍优于替换”，只能否定这些具体实现已获益。

来源：[直接四项](../osram_direct_input4_20261010/RESULT.md)、
[直接XCA/NPS](../osram_xca_direct_20261010/RESULT.md)、
[直接Nested容量8项](../osram_nested_direct_capacity_20261011/RESULT.md)。

## 四、已经做过的解释性对照

- **关系分支是否完全无用？** 同一训练后Relation checkpoint：R-off80.361，
  R-on80.576；旧Flat81.068。分支在该checkpoint有+0.216的条件贡献，但整体仍降。
  不能把整体失败说成分支从未发挥作用，也不能从此证明残差压制是根因。
  [证据](../osram_current_history_relation_20261003/residual_off_analysis/RESULT.md)
- **缩小Flat给图让容量？** Flat256三种子79.163→加Nested79.301，仅+0.138，
  seed67反而−1.001；仍低于大Flat80.559。
  [证据](../osram_small_flat_3seed_20261009/RESULT.md)
  中间Flat宽384/512/768/1024/1280、Nested MLP扩宽以及近等总参数预算也有已完成
  结果。Flat256+Nested704约13.561M参数，seed66仅78.152，未恢复大Flat81.068。
  [中间宽度](../osram_medium_flat_nested_20261009/RESULT.md)、
  [近等预算](../osram_nested_equal_budget_20261009/RESULT.md)
- **冻结原主路径只学Nested？** 同一个固定父checkpoint79.540→79.353。
  不与per-rate81.068混比。[证据](../osram_frozen_nested_20261009/RESULT.md)
- **图解码前是否包含可利用增量？** 同一冻结Nested的轻量Probe：Flat hidden79.654，
  拼Rooted79.059，拼PreDecoder78.396；这是三个Probe种子，不是三个主干种子，
  不证明信息论意义上的“没有信息”。
  [证据](../osram_nested_root_probe_20261011/completed/RESULT.md)
- **训练更久？** 匹配100@1e-3+50@1e-4的MOSI三种子：Flat80.747，Nested80.562。
  不能把150epoch与100epoch混排。[证据](../osram_complete150_20261011/RESULT.md)
- **改变训练监督？** Relation双读出、双视图task-only/InfoNCE已经做过。
  双视图rho.10的seed66为81.231，但三种子80.260，未复现整体优势。
  [证据](../osram_paired_history_rho025_20261002/THREE_SEED_RESULT.md)
- **MC-FWC** 已按用户要求取消；6完成/4中止/5未启动，只能按其独立验证集协议
  汇报部分结果，不得填出三种子完成均值。
  [取消记录](../osram_mc_fwc_20261011/CANCELLED.json)

四格、关闭Base/Gap/Local、历史删除、关闭不同delta等属于冻结推理诊断，
不能计作独立训练方法，也不能把局部纠错空间写成可部署成绩。

## 五、未完成与证据边界

- History-Innovation存在未完成seed，不列入完成三种子排名。
- Frozen Evidence Gate的80.741属于二阶段且best候选包含原Flat epoch0，不能混入
  一阶段排名或把其不降分保障当新模块收益。
  [证据](../osram_frozen_evidence_gate_20260930/RESULT.md)
- priority40最初报告主要是代码交付，37项共用64→1600残差接入、2项输入调制、
  1项归一化替换；不是40个互不重合的研究问题，也不能从代码报告推断40项都完成。
  [接入分类](../osram_priority40_20261004/RESULT.md)
- [Nested concat](../osram_nested_concat_20261011/STATUS.md) 已有启动和中途记录；
  本次未查远程最终产物，不把旧running文字当当前进程状态或完成结果。
- Hierarchical通道门不能改名成head路由实验；扩大直接GNN且保留Flat1600不能
  改名成“直接GNN与小Flat的联合等预算扫描”。
- Decision Correction同时改变读出结构、监督和有效容量，下降不证明决策校正
  原理必然无效；但足够说明它不是尚未尝试的新位置。

## 六、下一轮的去重规则（不是启动计划）

不能再把“Local条件化Memory”“显式当前—历史关系”“多头建图”“从残差改直接”
“分层决策校正”“缩Flat扩模块”当作未经尝试的新发现。
每个新方案先写清：相对最近历史版本，改变了什么可检验能力，哪个原方法不能
通过简单换名/换深宽实现同样职责，以及哪个已有对照能区分这个增量。
没有这种区别，就归为旧家族的参数/实现变体，不再打包成新研究方向。

因此本轮结论不是“读出侧不可能再提升”，而是：已在相同L/B/G信息上广泛尝试
重加权、交互、残差、直接映射与决策校正，尚无稳定整体优势。不能再把失败归咎于
“一直只加很小残差”，也不宜立即用另一种图/注意力/MLP名称开启下一批。
