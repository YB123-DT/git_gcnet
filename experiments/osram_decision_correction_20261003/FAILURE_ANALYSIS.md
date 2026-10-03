# 为什么新增模块没有形成稳定收益

INTERNAL DIAGNOSTIC ONLY

2026-10-03；证据版本 db78c30，decision 实现 2b00639。
本次仅分析已归档结果和代码，没有训练、新推理、权重修改或超参搜索。
使用结果分析与系统化诊断流程，区分可重复现象、实现事实和未验证解释。

## 1. 不是“完全没变化”，而是没有稳定正收益

相同 MOSI seeds 66/67/68；先平均八个 missing rates，再平均 seeds。
高缺失为 .5/.6/.7。全部沿用 per-rate BEST Test-oracle，不是正式
validation 选模成绩。以下不是所有历史实验的元分析，只比较近期两个完整变体。

| 模型 | 8-rate W-F1 (%) | 相对 Flat (pp) | High W-F1 (%) |
|---|---:|---:|---:|
| 原 cfg84 Flat | 80.559 | — | 75.594 |
| Gap-increment Filter | 79.579 | -0.980 | 74.699 |
| Evidence-centered Decision | 79.290 | -1.269 | 74.164 |

两个变体均三个 seed 全部低于对应 Flat。不能只归因于一个坏 seed；
也不据此进行统计显著性、泛化上限或整个研究方向无效的断言。

## 2. 新版的差距主要出现在哪里：配对 seed66

原模型来自固定 Full checkpoint 的分类输入干预：P_L 屏蔽 Base/Gap，
P_B 仅屏蔽 Gap，P_F 不屏蔽。新版来自同一 Full-selected checkpoint
的三个累计出口，不为各出口独立选 best。两边均使用原八率测试 masks。

| 路径／增量 | 原 Flat | 新 Decision | 新减原 (pp) |
|---|---:|---:|---:|
| Local | 74.820 | 75.678 | +0.859 |
| Local + Base | 79.439 | 79.411 | -0.029 |
| Full | 81.068 | 79.575 | -1.493 |
| Full - (Local + Base) | +1.629 | +0.164 | -1.465 |

高缺失 seed66 的 Gap 增量同样从 +2.487 缩到 +0.034。
这比“模型全部学坏”更准确：新版仍能利用 Base，但没有保住原模型的
Gap 增量。新版 Local 出口数值并未更低。

边界：这是两个分别联合训练模型的描述性比较；原低阶路径未接受独立
出口监督，而新版接受了。Memory 数值轨迹也不要求跨模型相同。
因此该表不是同一参数下的因果分解，不证明 centering、容量或 loss
中的某一项造成了差距，也不能用它评价独立训练的 Local-only 模型。

## 3. 诊断现象到有效模块之间缺少的证据

原 Gap audit 已显示 Gap 总体有用，但“存在 harm”不等于“推理时可以
识别 harm 并只抑制它”。seed66 中单变量、逐 rate 再宏平均的原方向
AUROC 为：Local margin .530、Base norm .552、Gap/Base ratio .537、
Base/Gap cosine .557、residual-query cosine/rho .517、eta .480。
这些简单量的总体区分度弱或不稳定；不证明完整高维表示没有可学习信号。

此前 Filter 的 seed68 在 .1–.7 selected checkpoints 上 mean g 为
.002–.011，表现为广泛抑制而不是已经验证的选择性去害。统计包含无
active Gap 的有效 utterance。其 rate0 在 Gap 增量严格为零时仍降分，
说明联合训练的主路径变化也不能忽略，不能把全部差距归因于测试时过滤。

## 4. 本轮并未单独检验 decision correction 的结构价值

代码与参数记录可以确认：

- 原 Flat 可训练参数 13,509,793；新版 3,992,487。新 head 433,799，
  旧读出 9,951,105 参数保留但不执行、不优化。这不是同容量对照。
- 目标从单一 Full task 改为 (Local + LB + Full task)/3。
  Gap 专属参数只受 Full 项监督，其显式系数为 1/3；Base correction
  参数受 LB 和 Full 两项监督，各 1/3。不能因此声称实际梯度范数
  就是 1/3 或 2/3，也没有梯度测量证明发生冲突。
- Centering 保证零证据时 correction 为零，不保证非零 correction
  有益。零初始化回到新 Local head，并非原 Flat 的功能等价起点。

所以“容量缩小”“早期出口监督改变了学习分工”“差分参数化不易优化”
均为候选解释，不是已经定位的唯一根因。结构、容量和目标同时变化，
限制了负结果的可解释性。代码测试和既有 mask/hash 检查降低了明显实现
错误的风险，但不构成所有代码均无 bug 或机制必然有效的证明。

## 5. 当前判断

保留原 Flat；这两版视为负结果，不继续自动扩大训练。
已有证据支持 Memory/Gap 有用，不支持“只要新增更复杂模块就会提分”，
也不支持“A/V 信息上限已经达到”。尤其不能把测试标签辅助发现的
可纠正样本，直接当作 Gate 在未见数据上可识别的样本。
后续任何候选都应先明确它新增的可检验假设，避免把现象描述直接升级为
机制结论，或同时改动多个因素后再追认单一原因。

## 来源与本次核验

- [本轮完整报告](RESULT.md)；`results/seed_{66,67,68}/metrics.json`、
  `results/baseline/seed_{66,67,68}/metrics.json`、`results/seed_66/PARAMETERS.json`。
- [原 Gap audit](../osram_gap_increment_audit_20261003/RESULT.md)；
  `../osram_gap_increment_audit_20261003/analysis/per_rate.csv`。
- [Filter 结果](../osram_gap_increment_filter_20261003/RESULT.md)及其三个 seed metrics。
- `gcnet_missing_m3/decision_correction.py` 与
  `gcnet_missing_m3/train_gcnet.py::_decision_correction_loss`。

本次重新计算各表数值；assert seed66 新旧 mask hashes 相同、原 audit
各 rate Full W-F1 与归档 baseline 相同。未重跑模型测试，也未将
重复 seed/rate exposures 视作独立统计样本；未做显著性或因果检验。
