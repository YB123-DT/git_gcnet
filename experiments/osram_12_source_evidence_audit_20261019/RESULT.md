# OSRAM 12项历史来源贡献分析

INTERNAL DIAGNOSTIC ONLY — 两组冻结审计均完成，未训练任何模型。

## 1. 协议与完成情况

- 不可变训练/推理源码快照：`cb02830e794297e9d3654b5f0c9a4900c6ac0cdf`。
  本次仅推理，不存在新增训练；沿用原始模型与分类器。
- MELD：seed66、原 lr1e-5/wd1e-5 Flat1600，固定原 miss0.3 checkpoint。
- IEMOCAP-6：seed66/fold5，固定原 miss0.3 checkpoint；不是五折平均。
- 两组各15条件：原 GCNet exact random masks 八率、Persistent 七种组合。
  共30/30完成，进程退出码均0，错误均为空，原参数 digest 不变。
- 原 Query、Local/Skip、联合 Block Write、Memory 轨迹均不变；TASR关闭。
  12项移除与8种来源组组合只重算原 Flat/LayerNorm/head。
- biggpu物理GPU1，UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`，
  cmerc-repro：Torch1.13.1+cu117/NumPy1.26.4/sklearn1.3.2。
- 两组并行；MELD107.15秒/峰值320.42MiB，IEMOCAP77.73秒/675.66MiB。
  时间包括加载、完整条件推理、压缩保存和统计，不是独占卡速度基准。
- 本地74项测试通过；实际远程环境24项审计测试通过。

**口径提醒：下面不是重训分数，也不替换已锁定的MELD三种子基线。**
这里始终用同一个miss0.3权重测所有条件，因此MELD Random8 ACC61.336
不能与原三种子、逐率Test-best的61.557直接混算。IEMOCAP也只有fold5。
两个源checkpoint均来自历史Test-oracle，结果不是独立测试泛化证明。

## 2. 来源分解是否正确？

以下是在全部真实批次/条件上观测到的最大绝对误差：

| 数据集 | Memory来源和 | Base来源和 | Gap来源和 | 重算原Flat logits |
|---|---:|---:|---:|---:|
| MELD |5.36e-7|3.58e-7|2.98e-7|0|
| IEMOCAP-6/fold5 |1.91e-6|1.91e-6|1.91e-6|0|

相对误差、原始Mask哈希、数据/权重身份、逐批诊断均保留在各数据集的
SUMMARY.json与REPORT.json。固定锚点在miss0.3的历史保存logits逐位复现；
全部随机率的Mask哈希与保护登记完全一致。前向及读出干预不消耗RNG。

读取来源在浮点误差范围内可加，但Flat是非线性的，12项分类影响不能相加。
精确三来源Shapley按每句话的负CE计算；效率误差均小于1e-12。
来源指联合更新中的Value槽位贡献，不是独立纯模态Memory或语义因果归因。

## 3. 原Memory是否真正影响分类？

同一个冻结模型，保留当前Local；“零历史读取”只把Base和有效Gap全部置零。
它不是单独训练的Local模型，也不是改写Memory状态。单位为百分数，差值为百分点。
各汇总均为对应条件的等权均值，Persistent6不包括ATV。

| 数据集/条件 | 完整ACC | 零历史ACC | 完整−零历史 | 完整UA | 零历史UA | 完整−零历史 |
|---|---:|---:|---:|---:|---:|---:|
| MELD Random8 |61.336|60.632|+0.704|38.861|36.877|+1.984|
| MELD Persistent6 |57.605|57.529|+0.077|32.041|31.096|+0.944|
| IEMOCAP-6 Random8 |62.616|44.871|+17.745|62.092|41.581|+20.511|
| IEMOCAP-6 Persistent6 |49.795|42.144|+7.650|49.899|38.589|+11.310|

| 数据集/条件 | 完整W-F1 | 零历史W-F1 | 完整−零历史 | CE降低量 |
|---|---:|---:|---:|---:|
| MELD Random8 |58.643|56.941|+1.702|0.047033|
| MELD Persistent6 |50.851|49.805|+1.045|0.036650|
| IEMOCAP-6 Random8 |62.383|41.638|+20.745|0.659670|
| IEMOCAP-6 Persistent6 |48.904|37.495|+11.409|0.360723|

MELD并非完全不使用历史：UA/W-F1/CE仍有收益，但Persistent平均ACC的净差很小。
IEMOCAP这一个固定checkpoint对历史读取明显更敏感。不能将这种响应差异直接
归因为写入失败、Query失败或特征质量；零历史读取也可能是分布外干预。
CE与ACC不必同向：例如MELD Persistent A的CE改善0.011744，但ACC从零历史
49.272降为完整48.889；不能只凭正CE收益宣称分类提升。

## 4. 来源贡献集中在哪里？

Random8的三来源组Shapley均值，收益函数为负CE，越大表示在这个固定游戏中
带来的平均CE改善越大。每个来源组同时包含其在Base和全部有效Gap中的项。

| 数据集 | 历史A来源 | 历史T来源 | 历史V来源 | 和：完整相对零读取的CE改善 |
|---|---:|---:|---:|---:|
| MELD |−0.003754|+0.046298|+0.004490|+0.047033|
| IEMOCAP-6/fold5 |+0.218402|+0.317135|+0.124134|+0.659670|

MELD的正向平均来源信用主要集中在T的Value写入来源，并不是三来源均匀有用。
这不表示每条Text都帮助、每条Audio都伤害，更不能据此用测试标签构造逐句Gate。
在完全没有历史Text的Persistent AV中，T来源状态与读取严格为零；并未补全Text。
不存在来源的Shapley可能出现约1e-9的舍入残差，不应当作真实贡献。

以MELD random miss0.3为例，三来源的写前状态平均Frobenius范数分别为
11.422/11.447/10.961，数值相近；但Base读出的来源范数分别为
1.158/2.664/1.259，决策价值也不相同。**保存的数值规模不等于读取规模，
读取规模也不等于分类价值。**范数不是可相加的概率份额。

同条件下Gap-T←T移除后CE升高0.008699，ACC降低1.434pp；原本正确变错误21条，
原本错误变正确9条。它在这837条Gap-T有效样本上有净分类帮助。
所有12项的有效N、纠错/致错及指标都在SLOTS.csv，不能把不同有效样本组直接
当成同一总体排序，更不能把12项独立影响简单求和。

## 5. 抵消是否解释了MELD的问题？

在random miss0.3的Base行，非零读取上的平均向量抵消程度：
MELD0.354，IEMOCAP0.400。MELD没有在这个条件下呈现比IEMOCAP更强的抵消。
这项观测不支持直接把弱分类收益解释成“MELD来源普遍严重抵消”，也不能排除
特定模态组合、具体样本或方向上的问题。完整四行、所有条件和分层均已保留。
向量抵消不是情绪冲突概率，不能对它赋予未经证明的语义。

## 6. 完整产物

- [MELD逐条件完整报告](MELD/RESULT.md)：15条件指标、每条件两张4×3表、
  存储范数、抵消和Shapley。
- [IEMOCAP-6/fold5逐条件完整报告](IEMOCAPSix/RESULT.md)：同上。
- 各数据集CONDITIONS.csv/SLOTS.csv/SHAPLEY.csv/SLOTS_GROUPED.csv：逐率、
  当前组合、历史ever-seen及coverage分层；inactive Gap保留N/A。
- [COALITIONS.csv](COALITIONS.csv)：两数据集所有条件×8来源组合，共240行。
- [COMPARISON.json](COMPARISON.json)：所有条件完整/零读取、CE收益、来源信用
  及Random8/High/Persistent6等权汇总。
- QUEUE_FINAL.json/EXPERIMENT.json和各SUMMARY/REPORT保留实际源码、数据、
  checkpoint、Mask、数值检查与资源信息。

原始逐样本来源向量、12项预测与8组合预测NPZ，以及每个分层的详细JSON，均保留
在biggpu独立输出目录，不上传模型/数据/大型预测到GitHub：
`/home/qinjinghui/remote_experiments/osram_12_source_evidence_audit_20261019/runs_v1/`。

## 结论边界

本轮回答了“来源贡献是否存在、被原Query读取、影响原分类器多少”，并确认
来源分解在浮点误差内成立；没有证明任何新Block有效，没有新增训练。
它没有将写入、读取、决策的潜在可改善空间完全因果分离。后续不自动加Gate、
改Missing或扩大训练/调参；也不把MELD的小净收益解释成历史完全无用。

## 补充：Gap-A不能根据miss0.3单点定性

已将全部八率12项的绝对ACC/移除差值整理为[ACC_PER_RATE.md](ACC_PER_RATE.md)，
Gap-A跨率及当前观测组合复核见[GAP_A_REVIEW.md](GAP_A_REVIEW.md)。
IEMOCAP-6中，“删Gap-A任一来源都提高ACC”只出现在miss0.3；删其历史A来源
在其他六个非零rate都降低ACC。该现象不能推广为Gap-A普遍有害，更不能据此
统一关闭该通道。不同rate的缺A目标样本不是同一批，跨率不能当严格配对干预。
本次补充只读既有CSV/NPZ，无新推理、训练、Mask或权重变更。
