# MOSI / MOSEI：全缺失率12项历史来源分析

INTERNAL DIAGNOSTIC ONLY。已完成，不训练、不改模型参数。

## 完整表格入口

| 数据集 | 八率 ACC：每率完整4×3 | 八率 W-F1：每率完整4×3 | 七种整段固定模态 | 纠错/致错与全部数值 |
|---|---|---|---|---|
| MOSI seed66 | [ACC](CMUMOSI/ACC_PER_RATE.md) | [W-F1](CMUMOSI/WF1_PER_RATE.md) | [Persistent](CMUMOSI/PERSISTENT.md) | [SLOTS.csv](CMUMOSI/SLOTS.csv) |
| MOSEI seed66 | [ACC](CMUMOSEI/ACC_PER_RATE.md) | [W-F1](CMUMOSEI/WF1_PER_RATE.md) | [Persistent](CMUMOSEI/PERSISTENT.md) | [SLOTS.csv](CMUMOSEI/SLOTS.csv) |

行：Base、Gap-A、Gap-T、Gap-V；列：联合写入中历史 A/T/V 的 Value 来源。
每格是**只在最终 Flat 输入处移除该来源之后的分类分数**，括号是相对同一行原完整预测的百分点差。
正差表示移除后变好，负差表示移除后变差。它不是 A-only/T-only/V-only 分数，也不是向量范数。
Base 使用全部有效句；Gap 行仅使用当前对应目标模态缺失的句子。
每行列出 active N、非零标签 polarity N 和同一批样本的原分数。未激活 Gap=N/A。
因此不同 Gap 行与 Base 的原分数不可直接混比，不可把12格独立变化相加。

## 固定口径，避免再次混淆

- 原 Flat，无 Router、TASR Adapter、Nested 或新增训练。原始 Local Skip、CHM写入/Query/读取和任务头不变。
- 两数据集均使用当前 gcnet-random-mask-v1。八率直接回放已保存 exact mask；Persistent 单独构造 A/T/V/AT/AV/TV/ATV。
- 每个数据集固定同一个 miss=.3 BEST：MOSI epoch105（原150轮 Flat）；MOSEI epoch19（原60轮 Flat）。
- **不是**八个 rate 分别换 BEST 权重。因此这些表中的全模型分数不等于此前八率 Test-best 总成绩，也不是新一轮重训下降。
- 只分析 seed66，不是三种子均值。没有使用正在运行的主干LR1e-5权重。
- ACC/W-F1：严格沿用 gold!=0、prediction>0。MSE 包含所有有效句，不套用分类 CE/argmax。
- 原 checkpoint 来自 Test-oracle，本轮为冻结行为干预分析，不是独立测试集性能证明。

## 便于先看的跨率对照：移除历史 Text 来源后的 ΔACC

单位：百分点；负数=删除使ACC下降。各列仍使用各自相同的行内样本，N在完整表中。

| Rate | MOSI Base←T | MOSI Gap-T←T | MOSEI Base←T | MOSEI Gap-T←T |
|---|---:|---:|---:|---:|
| 0.0 | +0.0000 | N/A | -0.7155 | N/A |
| 0.1 | +0.1524 | -4.8387 | -0.7430 | +2.7100 |
| 0.2 | +0.0000 | +0.6897 | -0.5228 | -0.9929 |
| 0.3 | -0.1524 | +0.9346 | -0.7705 | -0.9425 |
| 0.4 | +0.0000 | -2.6415 | -0.8531 | -1.8569 |
| 0.5 | -1.0671 | -1.2232 | -0.4953 | -0.7563 |
| 0.6 | -0.4573 | -1.0000 | -0.2752 | -0.4704 |
| 0.7 | -1.2195 | -2.4609 | -0.5504 | -1.7981 |

这张简表只截取历史 Text 两个位置；其他10格请看上方完整表，不用这两格替代全部来源结论。
MOSI Gap-T←T 在.3删除后提高0.9346pp，在.7删除后下降2.4609pp，显示冻结干预结果具有条件差异。
MOSEI .3/.7删除 Gap-T←T 分别下降0.9425/1.7981pp；同条件删除 Base←T 也下降。
这不证明两路互相独立，也不意味着历史Text具有纯粹的语义因果贡献。
小样本的低缺失 Gap 行尤其需要同时看N和纠错/致错数，不能只看百分点大小。

## 正确性与产物

共2数据集×15条件=30条件、360项来源移除。独立从原始NPZ复算每格ACC/W-F1/MSE及纠错/致错，
最大指标绝对误差2.842170943040401e-14。原完整读出重放误差为0；选定.3预测与原保存预测逐值一致。
MOSI来源读值求和最大绝对误差1.430511474609375e-6；MOSEI为3.814697265625e-6，均在既定浮点容差内。
权重摘要在执行前后相同，无梯度、无Optimizer或新checkpoint。

代码快照：1c5f29cb6c1592cdf0afe8874afc34a51ccd7938。
真实执行在biggpu物理GPU3，UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a；GPU4未使用。
MOSI全流程48.04秒，MOSEI336.53秒（含特征文件完整性检查和初始重放），非训练耗时。
详情见 [EXECUTION.json](EXECUTION.json) 和各数据集 FINAL_VERIFICATION.json。
完整状态/来源读取/逐句预测/全部分层在远程：
/data2/yb/remote_experiments/osram_scalar_12_source_audit_20261020/execution_v1/{CMUMOSI,CMUMOSEI}/。
大型NPZ不提交Git；原始checkpoint、mask和历史结果未覆盖。没有自动追加实验。
