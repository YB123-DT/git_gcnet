# MELD / IEMOCAP 实验分数、环境与可用性总账

核查日期：2026-10-16。**INTERNAL TEST-ORACLE DIAGNOSTIC ONLY。**

本次整理范围是当前讨论涉及的 MELD、IEMOCAP-4、IEMOCAP-6，以及 Flat、
B6、Nested、因果 ComP 和近期宽度/去 Skip/HPO 实验；不重新启动历史 MOSI
百模块搜索。所有分数均为百分数，差值为百分点。UA 是固定类别全集的
Macro Recall，不是 Macro F1。

## 0. 当前已暂停，不存在偷偷排队的新增训练

- 新 Flat67/68 的20个fold **未启动**，远程预定输出根目录不存在，无
  RUNNER/QUEUE，只有代码已提交并部署。此前 Flat66 的10个fold、B6的30个
  fold都已完成，不需要发送kill。没有终止、暂停或修改他人的GPU任务。
- OMX查询没有活动执行模式。准备好的三种子入口暂停，等待用户重新恢复。
- 保留所有旧结果和现有checkpoint，不删除“不好看”的分数。
- 详细暂停记录：`../osram_iemocap_flat_3seed_cmerc_20261016/STATUS.md`。

## 1. “可用”分三层，不等于“证明提分”

| 等级 | 含义 |
|---|---|
| A：当前可用 | 已完成且有对应口径证据，可报告为内部探索分数；只与匹配的环境、训练协议、种子范围进行比较。 |
| B：限定用途 | 历史环境、只有单种子、缺少同轮UA或训练协议不同；记录有效，但不能补齐不存在的证据或混进严格对照。 |
| C：不允许的用法 | 错配ACC/UA、把复测当Test-best、把单种子当三种子、把旧missing当新missing、把未运行当完成。这不是删除原始记录。 |

同环境还不代表逐样本共同mask、逐位复现、同一源码或只变一个变量。
当前所有Test-oracle/HPO结果仍是内部探索，不是独立测试泛化结论。

## 2. 环境核查

| 环境标记 | 实际路径/版本 | 证据边界 |
|---|---|---|
| C：cmerc-repro | `/data2/yb/conda_envs/cmerc-repro/bin/python`；Python3.10.13、Torch1.13.1+cu117、CUDA11.7、NumPy1.26.4、sklearn1.3.2 | 当前IEMOCAP Flat66/B6的manifest、实际命令、worker记录一致；MELD近期B6/匹配Flat亦有启动命令和Torch/NumPy记录。 |
| A：adapter历史组 | `/home/qinjinghui/anaconda3/envs/adapter/bin/python`；现在查到Python3.12.12、Torch2.9.1+cu128、CUDA12.8 | 原IEMOCAP Flat/Nested及旧MELD Flat/Nested的dispatcher明确记录此解释器。版本是现在查询值，非历史完整环境快照。 |
| U：历史启动环境未充分记录 | 不强行填C或A | ComP等记录不完整；IEMOCAP ComP脚本默认`/home/qinjinghui/anaconda3/envs/emo/bin/python`，但允许环境变量覆盖，故不能证明真实启动环境。 |

现在的emo环境是Python3.10.19、Torch2.9.1+cu128、CUDA12.8，同样不能倒推
它必然就是历史ComP启动时的软件版本。新入口增加了严格运行环境校验，
但这一保护不能追溯补全旧日志。

## 3. MELD：可保留的三种子结果

以下均为原GCNet随机missing、50epoch、八率0.0–0.7的Test-best ACC口径。
UA/W-F1来自同一个ACC选中epoch及其mask。宽度512只改变Adapter隐藏宽度，
output仍1600，保留Local Skip。No-skip仍保留Adapter内部Local。

| 实验 | 种子 | 环境证据 | ACC | UA | W-F1 | 可用范围 |
|---|---|---|---:|---:|---:|---|
| 接受的原Flat1600基线 | 66/67/68 | C，BASELINE及复跑记录 | 60.693 | 36.255 | 57.045 | A：保留原基线，不随HPO改变。 |
| Flat隐藏512 | 66/67/68 | C，启动文档 | 60.670 | 36.577 | 56.855 | A：容量实验；不是统一涨分，尤其高缺失UA下降。 |
| Flat matched（trial16全局超参） | 66/67/68 | C，实际命令/worker | 60.635 | 36.304 | 56.859 | A：下面两种B6的匹配对照，不冒充原lr=.001基线。 |
| B6 rank2/14，默认阈值及B6倍率 | 66/67/68 | C，实际命令/worker | 60.827 | 35.518 | 56.801 | A：可用负面证据，ACC小涨而UA下降。 |
| B6 rank2/14，调阈值/B6局部lr | 66/67/68 | C，实际命令/worker | 60.809 | 36.192 | 56.939 | A：没有同时改善平均ACC/UA。 |
| B6 training-HPO trial16 | 66/67/68 | C，实际命令/worker | 60.813 | 35.647 | 56.792 | A/B：配置筛选复验，有测试集选择偏差，非纯模块对照。 |
| B6 training-HPO trial13 | 66/67/68 | C，实际命令/worker | 60.691 | 35.877 | 56.945 | A/B：同上，不能忽略全局超参差异。 |
| B6 training-HPO trial12 | 66/67/68 | C，实际命令/worker | 60.706 | 36.000 | 56.806 | A/B：同上。 |
| 去Local Skip重训 | 66/67/68 | U，旧PROVENANCE缺完整环境 | 60.554 | 35.510 | 56.552 | B：同轮UA后来已精确恢复；不能继续沿用“UA完全没有”的过期说法，但启动环境尚未证实。 |
| 旧Flat GCNet-random复跑 | 66/67/68 | A | 60.498 | — | 56.754 | B：历史ACC/W-F1有效，不能填新Flat的UA。 |
| 旧Nested GCNet-random | 66/67/68 | A | 60.640 | — | 56.828 | B：可对比旧Flat的ACC；同轮UA和权重缺失。 |
| 因果ComP | 66/67/68 | U | 60.532 | 33.742 | 56.440 | B：Test-best分数有效，但每率分别训练，不是cyclic单模型模块对照。 |

**MELD matched对照的实际结论：**

- 默认B6减matched Flat：ACC **+0.192pp**，UA **−0.786pp**。
- 调参B6减matched Flat：ACC **+0.174pp**，UA **−0.112pp**。
- 因而“61.2说明第二模块稳定有效”不成立；61.2是选中的seed66，不是三种子均值。
- 原Flat1600的24个受保护权重仍在；matched Flat及两种B6共72个BEST保留。
- 七个新增matched训练均完成，六个worker在训练结束后的额外共同mask重放
  发生错误。Test-best已从选点artifact恢复并校验，故这些训练分数**可以用**；
  但失败的跨种子共同mask重放结果**不能声称已完成**。

### 3.1 MELD原Flat及宽度/去Skip逐种子

| 配置 | seed66 ACC/UA | seed67 ACC/UA | seed68 ACC/UA |
|---|---|---|---|
| 原Flat1600 | 60.675 / 35.436 | 60.603 / 35.830 | 60.800 / 37.498 |
| Flat隐藏512 | 60.766 / 37.359 | 60.330 / 35.961 | 60.915 / 36.411 |
| 去Local Skip | 60.560 / 35.404 | 60.517 / 34.783 | 60.584 / 36.344 |
| matched Flat | 60.781 / 37.107 | 60.321 / 36.161 | 60.805 / 35.645 |
| B6 rank2/14默认 | 61.216 / 35.935 | 60.560 / 36.011 | 60.704 / 34.606 |
| B6 rank2/14调参 | 61.193 / 36.008 | 60.445 / 37.193 | 60.790 / 35.376 |

宽度512三种子的总体UA比原Flat高0.322pp，但高缺失UA低1.256pp，不能描述
为整体稳健改进。宽度1024/2048只有seed66：分别ACC/UA为
**60.623/35.402**、**60.757/35.653**，只作单种子容量记录。

### 3.2 MELD六模块初筛：与三种子主表分开

环境C、seed66、50epoch，全部完成；保留Test-best及共同mask replay两个口径。

| 方法 | Test-best ACC | 同轮UA | 共同mask ACC | 共同mask UA |
|---|---:|---:|---:|---:|
| Flat同批对照 | 60.694 | 35.293 | 60.603 | 35.114 |
| B1 Head Graph Matching | 60.575 | 35.446 | 60.532 | 35.392 |
| B2 Dual Symbolic Attention | 60.781 | 36.131 | 60.560 | 35.888 |
| B3 Multimodal Bottleneck | 60.781 | 35.776 | 60.714 | 35.567 |
| B4 Neural Production | 60.661 | 36.189 | 60.469 | 35.808 |
| B5 TokenLearner/Fuser | 60.666 | 36.208 | 60.354 | 35.945 |
| B6 Learned Robust PCA | 60.690 | 36.915 | 60.407 | 36.915 |

这些单种子结果可以作筛选记录，不能和三种子均值混列排名。

### 3.3 B6近期结构/阈值搜索：有分数，不等于有模块贡献

环境C、seed66、每项50epoch，结构搜索20/20、阈值/局部lr搜索20/20均完成。

更早的全局训练超参搜索也已完成：B2和B6各20项，搜索lr、weight decay、
dropout，不是模块内部参数搜索。以下为该阶段的单种子记录：

| 全局训练HPO选择 | Trial | Test-best ACC | 同轮UA | W-F1 |
|---|---:|---:|---:|---:|
| B2最高ACC | 12 | 61.073 | 36.419 | 57.460 |
| B2最高UA | 1 | 60.757 | 37.766 | 57.609 |
| B6最高ACC | 16 | 60.915 | 35.767 | 57.175 |
| B6最高UA/注册目标 | 5 | 60.757 | 36.832 | 57.229 |

B2的注册共同mask目标最优为trial11，不是上表按Test-best ACC/UA挑出的
trial12/1；排序目标不同，不能随意更名为同一个“最佳”。上述B6 trial16
才是后面固定全局超参进行结构/阈值搜索的起点。

| 搜索与配置 | Test-best ACC | 同轮UA | 共同mask ACC | 共同mask UA |
|---|---:|---:|---:|---:|
| 结构trial3：rank2/14 | 61.216 | 35.935 | 60.900 | 35.879 |
| 结构trial11：rank3/8，同ACC并列 | 61.216 | 34.595 | 60.991 | 34.677 |
| 结构trial2：rank1/24，注册目标最优 | 60.934 | 37.421 | 60.800 | 37.188 |
| 结构trial19：rank3/10，共同mask ACC最优 | 61.116 | 36.155 | 61.121 | 36.148 |
| 阈值/lr trial11：ACC最优，后续三种子配置 | 61.193 | 36.008 | 60.886 | 35.840 |
| 阈值/lr trial8：注册目标最优 | 60.833 | 37.417 | 60.872 | 37.420 |

完整20项结构表见结构HPO RESULT，阈值搜索完整20项在远程SUMMARY/TRIALS。
不能把不同Study里的trial16或trial11当成相同配置。初期“调lr/wd/dropout”、
后来“调rank/steps”、最后“调阈值初始值和B6局部lr”是不同搜索。

结构搜索中的同checkpoint B6-on/off任务指标几乎不变，故最高模型ACC不能
自动解释为PCA分解在推理时发挥了显著作用。此类负证据也必须保留。

## 4. IEMOCAP：当前cmerc-repro组

100epoch、五折、GCNet随机missing、单模型cyclic八率，per-rate Test-best ACC。
以下UA均与ACC同轮，不采用最终随机复测UA。

| 数据集 | 模型 | 种子 | ACC | UA | W-F1 | 状态/用途 |
|---|---|---|---:|---:|---:|---|
| 4类 | 新Flat | 66 | 79.549 | 80.024 | 79.439 | A：完整五折单种子，80个Flat权重的一半。 |
| 4类 | B6调参版 | 66 | 79.408 | 79.655 | 79.263 | A：可与上行作同seed探索比较。 |
| 4类 | B6调参版 | 67 | 79.155 | 79.474 | 79.034 | A：该seed新环境Flat尚未跑。 |
| 4类 | B6调参版 | 68 | 79.298 | 79.564 | 79.184 | A：该seed新环境Flat尚未跑。 |
| 4类 | B6调参版 | 三种子均值 | 79.287 | 79.564 | 79.160 | A：B6自身三种子结果，不与单seed Flat当成等预算均值比较。 |
| 6类 | 新Flat | 66 | 62.011 | 59.945 | 61.382 | A：完整五折单种子。 |
| 6类 | B6调参版 | 66 | 61.637 | 59.819 | 61.239 | A：可与上行作同seed探索比较。 |
| 6类 | B6调参版 | 67 | 61.746 | 60.392 | 61.462 | A：该seed新环境Flat尚未跑。 |
| 6类 | B6调参版 | 68 | 61.818 | 60.147 | 61.350 | A：该seed新环境Flat尚未跑。 |
| 6类 | B6调参版 | 三种子均值 | 61.734 | 60.119 | 61.350 | A：同上。 |

保留情况：新Flat66共80个BEST+10份完整last_training；B6共240个BEST+
30份完整last_training。两者数值源码为785a012；B6包含明确的新增PCA配置。
新67/68入口仅修改种子准入和环境记录，不改数值模型，但**尚未启动**。

**同seed66比较（B6−新Flat）：**

| 数据集 | ΔACC | ΔUA |
|---|---:|---:|
| 4类 | −0.141pp | −0.369pp |
| 6类 | −0.374pp | −0.125pp |

现在不能声称B6在IEMOCAP提分。原先对旧Flat的微小提升，在这个新环境
单种子对照下未保住；完整新环境三种子Flat结果仍不存在。

## 5. IEMOCAP：旧adapter组仍有历史价值

同为100epoch、五折、原GCNet随机missing、cyclic八率、Test-best ACC。
旧Flat与旧Nested启动环境一致，可以保留它们自身的ACC对照；不能拿旧Flat
加上新Flat的UA，也不能与新环境B6组成严格环境控制实验。

| 数据集/模型 | seed66 ACC | seed67 ACC | seed68 ACC | 三种子ACC | 同轮UA |
|---|---:|---:|---:|---:|---|
| 4类旧Flat | 79.198 | 79.097 | 79.487 | 79.261 | 未保存/无法从现有旧权重恢复 |
| 4类旧Nested | 79.364 | 79.680 | 79.054 | 79.366 | 同上 |
| 6类旧Flat | 61.175 | 62.046 | 61.653 | 61.625 | 同上 |
| 6类旧Nested | 60.951 | 61.362 | 61.584 | 61.299 | 同上 |

旧Nested的30个fold实际全部完成，旧RESULT中的“仍在跑”已过期。旧Flat
240个BEST、旧Nested240个BEST被旧脚本主动清理，故不能承诺基于它们做
新的冻结诊断。已有最终随机mask复测预测不等于Test-best同轮预测。

在这组历史记录中，Nested对旧Flat：4类ACC约+0.105pp，6类约−0.326pp。
这是有正有负的历史结果，不是证明Nested普遍有效，也不需要把它们全部作废。

## 6. 因果ComP：分数有效，但不是同一种训练预算

ComP源码使用`argmax(test_accs)`选择epoch，随后直接取该epoch已存的ACC/UA；
没有再随机测试一次。故以下确实是Test-best，不因旧总表布局靠近“复测”
章节而改标成复测。MELD补W-F1重跑与原ACC/UA相符。

| 数据集 | seed66 ACC/UA | seed67 ACC/UA | seed68 ACC/UA | 三种子ACC/UA | 实际epoch/stage |
|---|---|---|---|---|---|
| MELD | 60.503 / 33.368 | 60.527 / 33.915 | 60.565 / 33.944 | 60.532 / 33.742 | 50 / 25 |
| IEMOCAP-4 | 75.445 / 75.950 | 75.138 / 75.467 | 75.196 / 75.485 | 75.260 / 75.634 | 200 / 100 |
| IEMOCAP-6 | 55.939 / 55.147 | 55.471 / 54.422 | 55.616 / 54.711 | 55.675 / 54.760 | 200 / 100 |

- ComP每个missing rate独立训练；OSRAM单次cyclic训练并独立选择八个rate的
  BEST。两者可以是各自协议下的baseline比较，不是严格相同训练方式的消融。
- 实际启动解释器记录不完整，**不能确认所有ComP都用cmerc-repro**。
- 已完成的IEMOCAP ComP记录是200epoch，不把它改写成150epoch。
- 没有找到ComP模型权重；IEMOCAP保留48份预测NPZ，但文件名不含fold，
  五折循环中被覆盖，实际只保留最后一折，不能把它们当240份完整fold预测。
  五折汇总日志可支持所报告的ACC/UA，不能支持新的全五折逐样本诊断。

## 7. 明确不能混用的东西

1. **Test-best ACC + 最终随机复测UA**：禁止拼接，必须同epoch、同mask。
2. **原Flat三种子 + 新Flat单seed66**：只能明确标注后并列，不能当同规模对照。
3. **adapter历史Flat + cmerc-repro B6**：保留历史参考，不作严格环境匹配提分依据。
4. **conversation-mask-v2 / gcnet-legacy-fixed-v1 + gcnet-random-mask-v1**：不是同一missing；
   旧分数留在历史记录，不进入当前GCNet-random主表。
5. **共同mask replay + 原Test-best**：都可能有效，但必须分列，不能互换列名。
6. **61.216或61.193单种子HPO赢家 + 三种子均值**：禁止混同；调参选择偏差要保留。
7. **旧Flat/Nested已有scalar日志 = 还有权重**：不成立；权重清理事件不能掩盖。
8. **ComP的最后一折预测文件 = 五折完整预测**：不成立。
9. **prepared代码/部署 = 已训练**：67/68的20个新fold尚未运行，已暂停。
10. **norm、probe可解码、branch-on/off几乎不变 = 模块一定有效**：均不成立。

这批记录没有理由整体删除。正确处理是保留有效分数，隔离协议/环境，
将缺失证据留空，撤回不受支持的“稳定提分”解释。

机器可读快照为同目录`SCORES.json`：55条模型/种子/汇总分数，另含B2/B6
全局HPO40项、结构20项、阈值/局部lr20项的状态、参数、Test-best及共同mask
均值。部分历史值只保留原报告三位小数；不可得值为null，不补零。输入JSON
的SHA256一并保留，避免再次把某个trial号与另一个Study混淆。

## 8. 源记录索引

仓库内路径均相对于`experiments/`：

| 内容 | 权威记录 |
|---|---|
| MELD原Flat1600，24个受保护权重 | `osram_meld_testbest_ua_20261014/BASELINE.md` |
| 去Skip同轮UA恢复 | `osram_meld_testbest_ua_20261014/RESULT.md`，不是旧“UA不可得”文字 |
| Flat512三种子 | `osram_meld_flat_width_20261014/MULTISEED.md`及SEED67/68_SUMMARY.json |
| MELD六模块单种子 | `osram_meld_block2_auto_20261014/SCREEN_SUMMARY.json`，Test-best和paired分别读取 |
| MELD三种子trial12/13/16 | `osram_meld_block2_topacc_3seed_20261015/SUMMARY.json` |
| 结构20trial | `osram_meld_b6_structure_hpo_20261015/RESULT.md`及SUMMARY.json |
| 阈值/局部lr20trial | biggpu `/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015/SUMMARY.json` |
| MELD matched Flat/B6 | `osram_meld_b6_top2_matched_flat_3seed_20261015/SUMMARY.json`，远程`testbest_recovery_v1/` |
| IEMOCAP新Flat66 | `osram_iemocap_flat_seed66_restore_20261016/SUMMARY.json` |
| IEMOCAP B6三种子 | `osram_iemocap_b6_transfer_20261015/SUMMARY.json` |
| 旧Flat/新Flat环境和轨迹差异 | `osram_iemocap_flat_seed66_restore_20261016/RERUN_DIFFERENCE_AUDIT.md` |

旧Nested/ComP真实完成记录在biggpu
`/data2/yb/paper/remote_runs_20261012/results/`：

```
osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_nested_s{66,67,68}/
osram_gcnet_random_mask_iemocap_nested_20261013/iemocap{4,6}/seed_{66,67,68}/fold_{1..5}/
comp_meld_causal_50e_3seed_20261012/causal/seed_{seed}/mr_{rate}/
comp_meld_causal_50e_3seed_wf1_20261013/
comp_causal_iemocap_3seed_200e_cpu4_20261013/{IEMOCAPFour,IEMOCAPSix}/seed{seed}/mr_{rate}/
```

本次只读取日志/配置/现有结果、整理报告；没有新训练、新推理、删权重、
改missing或改变任何已完成实验的分数。
