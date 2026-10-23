# OSRAM 全部 Block、调参和实验总账

INTERNAL DIAGNOSTIC ONLY

本轮只整理已有代码、配置、结果和归档，不训练、不推理、不读取权重内容，
不修改正在运行的任务，不删除历史记录。

## 先看哪张表

| 想看什么 | 文件 | 粒度 |
|---|---|---|
| 方便搜索的总表 | [ALL_BLOCKS.html](ALL_BLOCKS.html) | 四个视图：实验、运行、家族、机制入口 |
| 全部实验与家族 | [ALL_EXPERIMENTS.md](ALL_EXPERIMENTS.md) | 每个实验目录一行；附83条语义家族 |
| 每一个候选与配置入口 | [ALL_BLOCKS.md](ALL_BLOCKS.md) | 逐ID展开；包括106个Nested配置入口 |
| 可用Excel筛选的调参明细 | [RUNS.csv](RUNS.csv) | 数据集、seed、fold、维度、层数、Head、LR、WD、batch、选点与分数 |
| 全部逐率/逐条件指标 | [METRICS.csv.gz](METRICS.csv.gz) | 解压为CSV；原JSON叶级指标及JSON指针，不拼接不同评估 |
| 家族代表结果 | [FAMILY_RESULTS.md](FAMILY_RESULTS.md) | 带协议与出处；不是统一排行榜 |
| 注册表全部字段 | [BLOCK_CATALOG.csv](BLOCK_CATALOG.csv) | 机制、接入位置、实现文件、参数量、是否配置变体 |
| 原历史文档表格 | [SOURCE_TABLE_ROWS.csv.gz](SOURCE_TABLE_ROWS.csv.gz) | 解压为CSV；原表头、原行、文件与行号 |
| 完整运行配置 | [RUN_RECORDS.json.gz](RUN_RECORDS.json.gz) | 包含RUNS.csv精简显示之外的全部已提取配置 |
| 覆盖范围及统计数 | [MANIFEST.json](MANIFEST.json) | 本次实际扫描的本地/远端根目录与计数 |

## 家族导航

| 类别 | 已纳入内容 | 调整过的因素 |
|---|---|---|
| 原Flat与容量 | Large Flat、small Flat、memory-only adapter、Local Skip、分类/回归头 | Flat宽度、是否含Local、Skip开关、初始化、监督形式 |
| 历史强度与门控 | Memory-Shift、Post-GRN、总历史Gate、Local-conditioned evidence、Feature Gate、层次Gate、Gap-only Gate | scalar/channel/head、L2、宽度、深度、初始化、冻结/联合训练 |
| Current–History Relation | pairwise、普通容量对照、dual-readout、near-only splice、branch-off | relation维度、输出维度、接入位置、单/双读出监督 |
| 第一/二轮20方法 | meaningful20 round1与round2 | 各机制固定实现与单seed初筛；不把20个名字当共同结构 |
| Priority40 | M01–M40：集合/图、张量融合、调制、池化、特征交叉、几何、Mixer、静态原型等 | 机制与放置控制；保存实际代码路径 |
| New40与增量注册 | new40、meaningful_v3增量、结构/放置变体、撤回readout20 | 注册状态与实际训练证据分开；meaningful_v3不是后来的SourceRoleV3 |
| Core20与R系列 | C01–C20、实际实现的R02/R03/R12/R18及控制 | Memory/表示/概率/训练目标；提案不冒充实现 |
| Nested GNN | 原版、root/head拓扑、Local拆头、GIN/图控制、concat、直接替代、冻结Nested | dim、depth、MLP宽度、head、root、Flat宽度、decoder、residual scale、local LR |
| Neural Production等高分复查 | Neural Production、CWN、Perceiver IO、Tucker、XCiT-XCA及多seed | Local/Base/Gap更新范围、输入/输出残差、容量、接入位置 |
| 层次决策修正 | Local→Base→Gap、evidence-centered correction、Base-first Gap correction | 累计输出监督、共享反事实相减、Gap激活与无历史规则 |
| 历史直接读取 | B1近期历史检索、B2关系监督、短窗分支 | window、query/value、BCE、分支LR、原路径开关 |
| TASR与来源角色 | JointSourceTracker、来源中性/真实来源、SourceRole、Direct 2sigmoid | 冻结/一阶段、LR、单/双视图、残差/直接加权、gamma范围 |
| 来源关系与V3 | SourceRelation、SourceRoleTokens、HierarchicalSourceRoleV3、缩Flat扩V3 | 角色槽、同源pair、跨来源SA、角色CA、维度、头数、Flat宽度、decoder位置 |
| V3接法对照 | head-preserving、source-slots、Flat-hidden、topology full/diagonal、Local adapter/shared、pre/post Norm任务出口 | 一次只改变定义明确的连接因素 |
| 再查询与Teacher | conditional reread free/span、full-history teacher | 新地址vs原地址空间、直接历史访问、最终任务监督；尚无成绩的不填写 |
| 训练目标 | paired-view task-only、InfoNCE、rho扫参、JEPA、Future-State、MC-FWC | view2比例、drop、辅助权重、梯度路径、取消/smoke状态 |
| 原Memory超参 | write_step、ridge、alpha/beta、head维度与数量、Gap residualization | 冻结敏感性与从头训练分开 |
| 原优化器参数 | LR、WD、cosine、100/150epoch、batch、fixed-rate/cyclic | 改训练配置不计为新Block；固定rate的8模型不等于cyclic的8个BEST |
| 数据与协议 | ECERC、MM-Align、MAGTKD、MMGCN、GCNet missing更正、ComP复现 | feature hash、环境、mask、划分、选点，均不可跨口径混排 |
| 冻结分析 | Local/Base/Gap、12来源、Query/Addressing、历史删除、Retention、Probe、梯度 | 不计为新端到端Block；Oracle与消融成绩不能当模型训练成绩 |

## 计数与成绩边界

- “实验目录”“机制入口”“配置变体”“运行证据”是不同计数。目录中可能只有计划、诊断或汇总；入口也可能只有代码。
- 同一训练轨迹的8个per-rate BEST不是8次训练；不同seed、fold、trial是不同配置/运行。
- 同配置在本地、远端、归档可能有副本。保留来源，不把副本数量称为独立训练次数。
- `RUNS.csv`保留叶级结果及嵌入式Optuna/归档记录；`same_config_evidence_count`是重复证据提示，不是自动合并依据。
- 83条是导航用语义家族，不代表83项科学创新；311条是注册/配置条目，不代表311项已训练方法。
- `conversation-mask-v2`、`gcnet-random-mask-v1`、`gcnet-legacy-fixed-v1`和`unknown`分开。只有显式配置或已核验执行源码hash才确认Missing。
- Test-best、共同Mask重放、Persistent、validation选点、oracle拼接和probe不能填进同一个排名。
- 旧trainer的`selected_*_by_rate`与最后随机重测可以不同。能读到原选点字段时优先使用，原重测单独放`FINAL_RETEST_*`；没有同次UA就不填。
- ACC/UA/W-F1统一为百分数；UA仅接受明确Macro Recall等字段，绝不以Macro-F1替代。
- 只有完整8率才计算叶级八率均值。缺UA、环境、完成状态或源码证明时不猜测、不补造。
- `complete_declared`表示记录声明完成；不是本轮重新检查过全部checkpoint。旧启动文档和注册声明可能过期，以后续完成结果为证据。
- 家族代表分数只是导航，完整明细以原文件/JSON指针为准。所有未统一解析的旧表格仍保存在原文表格索引。

## 复现盘点

```bash
python -m unittest discover -s tests -p test_all_block_inventory.py
python -m experiments.osram_all_block_inventory_20261023.inventory --include-remote --refresh-selected
# 已有完整本地扫描快照时，无需再次访问服务器：
python -m experiments.osram_all_block_inventory_20261023.inventory --rebuild
```

`SCAN_SNAPSHOT.json`是可再生成的扫描中间件，不上传Git；不能凭缺少它判断原实验不存在。
机器可读CSV、MANIFEST、源文件hash及家族JSON上传，大指标表和完整运行JSON以gzip无损压缩上传；
运行脚本同时保留本地未压缩CSV。原权重、数据、预测数组均不上传。
原始扫描只读既有实验根目录；明确跳过源码快照、数据、环境和权重内容。

## 已知限制

本表覆盖MANIFEST明确列出的仓库与远程目录，不能证明所有服务器和已彻底删除且无任何保留记录的实验都存在于表中。
扫描失败、超过读取上限或无法解析的文件单独列入`SCAN_ISSUES.json`，不是将这些实验判为失败。
未归类目录仍完整列出，不因为暂时缺少语义家族标签而丢掉。
