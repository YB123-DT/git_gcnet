# 训练、输入、Memory与协议实验补充台账

2026-10-23；只读现有源码与归档，不启动训练、推理、远程写入或新诊断。机器可读详情见 `training_input_families.json`。这是全量目录总账的语义补充，不替代逐run结果清单。

## 必须分开的五类

1. **推理Block**：例如CED输入块；模块清单由另一条审计线覆盖。
2. **训练策略/损失**：双视图、InfoNCE、JEPA、MC-FWC；多数不增加推理Block。
3. **超参数**：LR/WD、轮数、Flat宽度、Head/KV、write step/ridge；不同配置不是不同方法。
4. **协议/数据**：missing sampler、环境、特征来源和选点方式；不能算创新Block，更不能混排分数。
5. **冻结诊断**：读出消融、history deletion、source/Shapley、probe；probe虽训练轻量头，也不等于重新训练原模型。

## 语义家族总表

分数仅为明确原报告中的代表项，不声称每项均完整覆盖所有seed/rate。W-F1、ACC、UA单位为百分数，差值为百分点。所有Test-oracle结果仅内部探索。

| ID | 家族/接法 | 调过什么 | 已有证据状态 | 代表结果或限制 |
|---|---|---|---|---|
| T01 | 双视图task-only | View2 drop=.2，rho=.5 | MOSI66/100轮完成 | Random W-F1 81.068→80.386；Persistent六组合67.639→71.583；推理单view |
| T02 | 双视图InfoNCE | lambda=.1/temp=.1、rho=.5/.1 | 两个MOSI66配置完成 | rho.5 W-F1 80.267；rho.1 80.191，分别低于对应task-only |
| T03 | 双视图rho loss权重 | .05/.1/.15/.2/.25/.3/.35/.4/.45/.6；.5旧对照 | 10配置66完成；.05/.1/.25补67/68 | rho.1单seed81.231，但三seed80.260低于Flat80.559 |
| T04 | 双视图MELD/IEMOCAP6迁移 | 固定rho.5/drop.2 | MELD66与IEMOCAP6五折66，共6job完成 | MELD Random ΔACC−.211/ΔUA+.552；IEMOCAP6−.278/−.171；不是所有条件收益 |
| T05 | MC-FWC | horizon/tau/lambda/trajectory；E0–E4+shuffle | 实现/69tests/smoke完成；训练被用户取消 | 6完成、4停止、5取消；验证集单checkpoint协议，不与旧Test-best硬比 |
| T06 | Complete-State JEPA | state目标/EMA/预测器预算 | 五seed完成 | W-F1 State79.890 vs No-JEPA80.076 |
| T07 | Write-State/WSC JEPA | 写后state监督/匹配预算 | 五seed完成 | WSC79.787 vs Joint80.002 |
| T08 | Future-State JEPA | post-write query→下一句EMA Local，weight.1 | 实现+真实GPU smoke | **无正式训练F1**；不是双视图InfoNCE |
| T09 | 监督Frozen Teacher | EMA→预训练冻结projector targets | 原报告工程smoke；后续runner引用已训练Teacher | 不能用早期“未训练”文档否认后续训练，也不能凭引用补造F1 |
| T10 | Text predictable subspace | R32/Q/C、variance/covariance/sentiment约束 | 有界两次更新smoke | **无完整五seed分类结果**；低rank不等于训练坍塌 |
| T11 | 三数据/MOSI-only M3预训练 | 数据混合、更新预算、9 directed routes | 五seed表征预训练完成 | MOSI val centered cosine .1019→.1433，**不是情感F1** |
| T12 | 旧B2 source-only completion | Stage1+Stage2、read-node reinjection | 五seed100+100轮完成 | per-rate 79.758 vs P0 80.002；同目录另一套单epoch79.077 vs79.441，不混用 |
| T13 | 预训练completion双projector/冻结回灌 | projector bank、freeze/init/reinjection | dual-projector三seed完成；其他独立记录 | W-F1 80.559→78.842；预估模态不进入原persistent写入 |
| T14 | reg-only JEPA/fullMMoE/simple-reg | joint-reg-only/emotion-only、Teacher、cfg84容量 | 多个runner/引用；无一个统一最终总表 | 与readout Block不同，不把缺本地汇总写成“未跑” |
| T15 | 情绪任务头/损失 | binary/threeclass；MSE+BCE | dual-loss五seed完成等 | 旧协议W-F1 80.08→80.14；分类任务变化单独标记 |
| T16 | 训练missing分布 | forcedText .25；cyclic去miss0 | 五seed完成 | forcedText79.74 vs80.08；去0为80.038 vs80.445 |
| T17 | CED输入Möbius分解 | 单/双/三阶槽、子集共享dropout | MOSI66/100完成，+398086参数 | W-F1 79.875 vs81.068；实际替换OSRAM输入node，不是仅分类侧Local |
| T18 | One-stage Full-History Teacher | causalTransformer3×512、final-only MSE，约11M | 计划与独立模型代码存在；目录无完成结果 | 这是替代历史架构，不是OSRAM第二Block；不报告已完成 |
| T19 | 早期OSRAM容量 | Heads/K/V/output/causal/forward-slot复用 | 多个历史实验记录 | 不能与新GCNet missing直接混排 |
| T20 | 90规格优化/容量筛选 | LR/batch/Adam(W)/WD/dropout/LR倍率/clip/schedule/epoch/容量 | 90为规格计划；现本地是seed66 partial60行 | partial cfg01=80.828，不能宣称90全部完成或90个独立Block |
| T21 | Flat大小/2400/小Flat等 | 隐藏宽、output、预算、Nested容量分配 | runner与多个完成JSON | 是配置/容量对照，不是新机制计数 |
| T22 | 100→150完整续训 | constant150或100轮1e-3后50轮1e-4 | complete150 47完成、15 IEMOCAP4 Flat取消 | 旧MOSI Flat80.747/Nested80.562；不混新GCNet MOSI |
| T23 | 低LR/batch64 | 1e-4；batch64对应2e-3 | 旧screen完成；新MOSEI Flat67/68另表 | 旧MOSEI83.787/83.939/84.020不能直接对比新miss67/68 |
| T24 | MOSI新miss LR定版 | 1e-3/5e-4/3e-4；101轮降10倍 | 3e-4补三seed完成 | **当前Flat W-F1 79.2817，高缺失73.4715**；66=79.2491 |
| T25 | MELD Flat LR/WD | lr2e-4/1e-4/3e-5/1e-5；WD0至1e-3 | 多组与三seed完成并登记 | **当前baseline ACC61.557/UA37.728/WF1 58.509**，lr1e-5/wd1e-5 |
| T26 | MELD余弦/写入.3 | warmup .05+cosine、step.6/.3 | 三seed完成 | step.6 ACC60.749/UA36.781；step.3 60.611/36.474，不支持统一提分 |
| T27 | 独立rate训练 | 每rate单独模型 vscyclic | 新MOSI8+MELD8模型完成 | MOSI fixed78.681 vscyclic79.249；MELD fixed60.886/37.069，非persistent |
| T28 | MELD写入强度/ridge | step.3/.6/.9；ridge1e-4/1e-3/1e-2 | 五配置66/50轮完成 | W0ACC60.838/UA35.633；step.3 60.948/35.653；ridge.01 60.560/36.717 |
| T29 | MOSI写入/head维度 | eta.2/.4/.6/.8/1；KV32/48/64/96/128；48补三seed+.4 | head48三seed完成；其余以各队列为准 | 64→48参数13.510M→11.246M，W-F1 79.282→79.293；并非稳定提分 |
| T30 | 保留/alpha-beta/head冻结敏感性 | 参数覆盖、write interventions、head leaveout | 多份冻结报告 | 训练后alpha约.974–.976、beta约.499；覆盖实验不是训练最优设置 |
| T31 | 早期step1→.6/query参数 | write_step、Gap residual、query availability、history support | 历史runner与早期启动记录 | 不根据旧“RUNNING”文件判断今天远程进程；不填未核实最终分数 |
| T32 | MELD特征切换 | ECERC/MAGTKD/MM-Align/MMGCN | MAGTKD/MMGCN训练完成；MM-Align下载不完整未训 | ACC/UA：MAGTKD54.631/31.942，MMGCN57.826/30.562；同数据集不同特征 |
| T33 | missing/环境/选点纠正 | random-v1 /legacy-fixed/conversation-v2，ACC同次UA恢复 | 原分数与更正记录均保留 | 旧60.498不能填新UA；历史复跑baseline60.693/36.255，后续锁定61.557/37.728 |
| T34 | IEMOCAP4/6新环境Flat | 五折×三seed、cmerc一致性 | 30fold训练完成 | IEMOCAP4 ACC79.351/UA79.632；6类61.714/60.101，100轮 |
| T35 | ComP causal/noncausal | 外部baseline、逐rate训练、dataset/seed | 历史总账明确结果与环境限度 | MELD causal3seed ACC60.532/UA33.742；不是OSRAM接入Block |
| T36 | OSRAM前的Missing-M3阶段 | JEPA/LR/WD/任务/窗口/冻结补全/两阶段等 | 多目录历史实现与结果 | 计入所有实验目录，但不计作当前OSRAM第二Block或新方向 |
| T37 | Frozen probe/history/gradient审计 | probe、donor/history删除、梯度小步、视觉零特征 | 多目录独立诊断 | Probe训练不等于原模型端到端Block；旧MELD Probe是legacy-fixed |
| T38 | Persistent/LocalBaseGap/scale推理 | 组合保留、读出zero、历史alpha、oracle | 冻结推理/离线统计 | 不计新训练；oracle分数不可部署；constantpattern与randomrate分开 |

## 数字去混淆

- **旧MOSI80.7**：`osram_complete150_20261011` 的旧missing/旧训练记录；不是新GCNet Missing下79.28的同一条件。
- **当前MOSI**：`osram_mosi_mid_lr_3seed_20261021/completed/RESULT.md`，新GCNet、s0、150轮3e-4→3e-5，三seed W-F1 79.2817。
- **MELD60.693**：lr1e-3历史受保护Flat三seed；**61.557**是后来用户锁定lr1e-5/wd1e-5的三seed。不能把两个baseline混在Block排名。
- **61.786/38.435**与**61.786/38.420**：前者wd2e-4的O6单seed，后者当前wd1e-5 baseline seed66。ACC四舍五入相同不表示同一个实验。
- **B2**至少重名三次：旧source-only completion、六模块筛选的Dual Symbolic Attention、近期历史R/S设计中的关系版本。必须用机制和路径，不仅用B2标识。
- **TASR的Persistent双视图**与旧paired history不同：前者从完整训练数据独立造persistent view；后者View2只对View1再删。TASR家族在模块清单详列，此处不重复计方法。
- **Test-best ACC对应UA**和**UA-best选点**不是同一成绩列；**共同mask replay**与原始stochastic Test-best也必须分开。

## 完成状态证据边界

没有在本次连接服务器或核实实时PID。上述状态以指定本地报告为准；已完成、smoke-only、取消、数据缺失、仅计划分别保留。机器JSON为每家族记录明确路径/来源，便于主总表与逐run目录做join。没有因为一个早期README仍写“未启动”就覆盖后续RESULT已经完成的事实。

这次整理不会把用户最新撤回的LISTA输入想法或OTW写入建议当作已实现/已训练实验；也不会把讨论中的公式增加为“方法总数”。
