# Readout / Source / Nested 语义归档补充

只读整理，未启动训练、推理、调参或修改模型。此文件覆盖 45 个语义家族、138 个不重复实验目录引用；它不是整个仓库的实验总数，也不能把一个家族、一条配置、一个 trial、一个 seed/fold 各算成同一个单位。完整逐运行事实由主清单连接。

## 所有本 lane 家族

|家族 ID|Block / 内容|实际插入位置或职责|调过的因素 / 变体|目录引用数|
|---|---|---|---|---:|
|legacy_fusion|早期 Local/历史融合替换|LocalCrossAttentionFusion / LocalCenteredContextFusion；替换旧融合，不等同cfg84后置残差|local-cross-attn；local-gated；local-centered；modality-tracks|4|
|history_innovation|History–Innovation插值读出|历史与当前映射后scalar K插值；当前注册已移除，只有旧审计引用|history-innovation（取消的历史归档）|1|
|memory_shift|Memory-Shift-Residual / scalar filter|Relation128；四证据共享scalar filter；零初始化映射加Flat pre-norm|D1-W128；D2-W128；D2-W256；D3-W128；D3-W256|2|
|post_grn|Flat后GRN|x=Flat hidden，condition=读出消融后的Local/Base/Gap/a；channel gate×value加x|post-GRN开/关|1|
|history_input_gate|总历史scalar Gate|alpha=1+0.2tanh，统一缩放Base/三个Gap|one-stage joint|1|
|history_query|历史支持条件化Gap Query|Query adapter，不是读出Gate|random-missing history-support query|1|
|evidence_gate|Local-conditioned scalar evidence Gate|每证据单scalar，中心1，L2保持正则|Base+Gap四gate；Gap-only；L1约束|3|
|frozen_evidence_gate|冻结Flat二阶段Gate|3seeds×8父checkpoint，各训练gate；与一阶段不能混排|epoch0；oracle-best含epoch0；final100|1|
|hierarchical_gate|Feature Gate＋Evidence Softmax|每路内部feature/head过滤再active evidence间竞争；K×softmax|feature＋evidence two-level|1|
|feature_gate|仅8×64 Feature Gate|无evidence-level softmax竞争|8×64 feature-only|1|
|local_skip_gate|Local Skip Gate|只门控完整LocalSkip输出|scalar/channel配置以run.py为准|1|
|memory_only_adapter|Adapter不含Local|保留原LocalSkip；Adapter只接Base和Gap|memory-only adapter|1|
|current_history_relation|Current–History Relation|共享投影+逐证据[q,k,q*k,/q-k/,type]→mean→Flat残差；非Gate|pairwise relation；同容量pooled control（仅实现未训练）；0.5base+0.5full任务监督；near-only splice（离线）；relation-off（冻结）|2|
|gap_increment_filter|Gap增量scalar filter|同轨迹uFull-uBase；LN(uFull+(g-1)increment)|3seeds|1|
|decision_correction|Evidence-Centered Hierarchical Decision|新Local决策+中心化Base/Gap决策差值；三个出口各1/3任务监督；旧Flat不执行|Local；Local+Base；Local+Base+Gap|1|
|base_first_gap|Base-first Gap-correction|u=Skip(L)+FB(L,B)，Gap后修正u；不是原Flat保留|Base-first Gap correction|1|
|counterfactual_readouts|多候选/边际读出|冻结或新训练的边际组合路径；需区分是否单独头|原Flat历史recency元数据；counterfactual候选|2|
|nested|原始Nested GNN及结构消融|Rooted子图GIN，解码修改Adapter输入Local/Base/Gap；Skip保持原Local|old dim64 depth3 groups8；plain-GIN；no-markers；no-head-edges；last-layer；dim32/128；depth1/2；groups1/4；root-aware pooling；Local8；groups1dim512|4|
|nested_capacity|Nested容量/Flat参数重分配|扩大图节点宽度或GIN MLP；缩小Flat；必须同时记录Flat/Nested容量|GIN MLP256/512；Flat256+Nested704；Flat widths384/512/768/1024/1280|5|
|small_flat|Flat容量 / 小Flat+模块|只改emotion_adapter hidden width；原output_dim1600/Skip不变|Flat256；Flat256+NPS；Flat256+Nested64|2|
|nested_direct|Nested直接替换Adapter证据输入|去掉输入residual，不删除Flat；zero/random decoder初始化分开|direct-zero64/64；direct-random64/64；64/128；64/256；64/512；128/128；128/256；128/512；256/256；256/512|4|
|nested_training|Nested训练/LR/epoch/跨数据集|原图模块不同优化/训练预算，不能当新机制|100/150epoch；constant/cosine；局部LR+warmup+randomDecoder；MOSEI dim32+depth2；MELD d32-l4-g2-m64等|11|
|nested_gate|Nested解码残差Gate/固定系数|只修改Nested输入校正强度，不能描述成CHM Gate|allGap Gate；GapT-only Gate；fixed residual scale grid|3|
|nested_concat|Nested Rooted concat独立读出|保留rooted表示与Flat表示concat/Probe，不是相同残差架构|concat；frozenNested-only学习；frozenRepresentationProbe|3|
|nested_diagnostics|Nested只读消融/梯度|无新训练；不是调参方法数|单槽关；全部/组合关；输入梯度；训练梯度；四数据集逐rate汇总|6|
|nps|Neural Production|四规则共享多步选择/更新；原版Base/Gap输入残差，Local改版独立|Base/Gap-only四规则三步width96；Local+Base/Gap width96；Local+Base/Gap width256|2|
|direct_other_modules|多机制去输入残差|XCA / NPS /其他输入adapter direct不是全部删Flat|XCA direct；four direct input candidates|2|
|related_history|B1 R / B2 R+S短窗历史|历史Local短窗8；B1检索；B2额外训练关系BCE及same/change拆分|B1输入2435；B2输入2564；B2关系loss0.05；分支off/S=0.5（诊断）|4|
|tasr|TASR Gap来源轨迹校准|JointSourceTracker精确Value槽来源分解；Gap9路 bounded0.25tanh加原Gap|F0；F1 source-neutral=Gap/3；F2 source-real；adapterLR3e-6/1e-5/3e-5/3e-4；冻结双视图0.8/0.2；冻结random-only；joint random-only|4|
|text_path_router|BaseText/GapText候选Router|Full/去GapT←T/去Base←T三候选预测，训练概率混合或标量混合；不是TASR|frozenRouter；jointRouter；主干1e-5/2e-5；IEMOCAP6五折；MOSI新Miss/选点修正|10|
|coupled_source_role|Base-Gap Partner Source Role|12来源角色共享MLP；原版原读出+0.25tanh权重delta；direct版2sigmoid来源重构|residual MOSI/MOSEI；direct MOSI3seed；direct MOSEI/MELD/IEMOCAP6；gamma=1冻结|5|
|source_relation_v1|SourceRelation Base/Gap均值包|每来源Base与masked-meanGap交互→三个128tokens→SA→LocalCA→全局1600残差；原提示明确含mean|MOSI150；MOSEI60|1|
|source_role_tokens_v2|12-source-role tokens重做|不提前meanGap；12个128tokens Attention→全局Flat残差；用户取消未完成MOSEI|MOSI完成150；MOSEI20后取消|1|
|v3|V3 Hierarchical Source–Role|同源Base-eachGap343pair→同role跨sourceSA→Local roleCA→四零初始化64→512decoders回填Base/Gap|MOSI seed66/67/68；MOSEI；MELD；IEMOCAP4/6五折；classification调度+3seed补充|3|
|v3_capacity|V3宽度/Flat宽度协调|缩小Flat hidden、扩大V3width；容量对照同协议|Flat512/1024；V3width128/256；Plain匹配control；MOSI f512v128三种子|4|
|v3_direct|V3非残差读出替换|V3 decoder输出直接作为Flat Base/Gap输入；不再加原read|MOSI direct；MOSEI direct role|1|
|v3_connections|V3三种连接位置|保head回填/来源槽/Flat隐藏层接口|head-preserving；source-slots；flat-hidden|2|
|v3_paired|V3六组成对接口|来源连接拓扑；更新Local与Skip一致性；任务增量Norm前后|topology-full；topology-diagonal；local-adapter；local-shared；task-pre-norm；task-post-norm|1|
|reread|Conditional Memory Reread|首读Local/Base/a生成第二query，再读同一alpha衰减prewriteM；Base校正|free-query；original-effective-query-span|1|
|evidence_audits|历史来源/组合/Query决策冻结诊断|12路消融+8来源Shapley+BaseTextGapText交互；不是新模块或正式训练|Local/Base/Gap组合；12格移除；3source Shapley；query vs readout frozen probe|12|
|reporting_audits|历史读出/高分审查|已有结果证据整理，不增加训练|placement review；historical shortlist|5|
|meld_b3|MELD B3 MBT瓶颈局部/HPO|同Block2输入adapter；bottleneck数/交换轮数与局部参数搜索|tokens1..8；rounds1..4；默认4/3；局部宽度/学习率参数按trialconfig|5|
|meld_b6|MELD B6 Learned Robust PCA局部/HPO|unrolled low-rank/sparse evidence decomposition输入校正|rank；迭代步数；阈值初始化；B6独立LR；training lr/wd/dropout|5|
|meld_block2_screen|MELD六Block+同协议Flat筛选|统一MeaningfulInputAdapter路由；不是六种完全新机制|B1 graph_matching_base_gap_pairs；B2 dual_attention_symbolic_relations；B3 mbt_role_bottleneck；B4 conditional_new_07_neural_production；B5 tokenlearner_fuser_v11；B6 learned_robust_pca_evidence_decomposition|1|
|meld_flat_width_skip|MELD Flat宽度/Local Skip结构对照|改变Flat adapter容量/去LocalSkip重新训练及冻结证据消融|adapter512等宽度；LocalSkip on/off；BaseGap readout combinations（诊断）|6|

## 必须避免的历史混淆

1. **Nested direct 不是删除 Flat。** `TokenAdapter(residual=False)` 移除的是原 Local/Base/Gap 与图解码输出的相加，后面的原 `emotion_adapter/local_skip/emotion_norm/task head` 仍执行。direct-zero 与 direct-random 的初始化不同，不能混为一组。
2. **XCA / Perceiver / Tucker 并不是以 Flat1600 hidden 作为核心输入。** 它们从 Local 和有效前向 Base/Gap 构造证据，输出加在 Flat pre-norm anchor。不得将“相加位置在 Flat 输出”误写成“模块输入来自 Flat hidden”。依据：`experiments/osram_readout_path_audit_20261009/RESULT.md`。
3. **原 Nested / CWN 校正 Adapter 输入，不校正 Local Skip。** 原 NPS 只修正 Base/Gap；NPS+Local 改版才增加 Adapter 的 Local 校正。
4. **旧 mean8=80.559 / seed66=81.068 与新 GCNet-mask MOSI seed66=79.249 不是同协议基线。** 本 lane 不将这些绝对分数跨协议排名。确切 Missing sampler version 必须来自每次有效配置，不能靠日期猜测。
5. **History-innovation 已不在当前注册表。** `osram_readout_path_audit_20261009/RESULT.md` 记载它的取消归档：66/67 已完成、68 在96轮停止；本 lane 未找到当前文件夹或可验证最终分数，不能给它伪造一个完成成绩。
6. **SourceRelation 最初的 Gap mean 是原提示明确写出的结构。** 后来12-token重做是设计修订，不应把最初实现描述为没有按含 mean 的原公式做。12-token MOSEI 被用户在20轮取消，不能作为完成60轮成绩排名。
7. **V3 的原 RESULT/三种子 RESULT 是启动快照。** 目录名、README 标题或旧 RUNNING 字段不能替代更晚的 completed/REPORT、完整 history 与预期产物。MOSEI grid 的标题虽然说 completed，正文四个新增 arm 仍为 running；不能只读标题。
8. **`osram_mosei_direct_role_readout_20261021` 是冻结 L/LB/LG/Full 消融，不是新增非残差训练。** 本清单已归到诊断而不是 V3 direct 架构。
9. **TASR F0/F1/F2 冻结初筛、TASR joint、12路 CoupledSourceRole、2sigmoid direct、三候选 Text Router 是不同方法。** TASR 首轮固定单一 miss0.3 原 checkpoint，不能将它与各 rate 独立 Test-best 的基线直接相减。
10. **F1=每来源使用原 Gap/3，F2=真实 Value-column 来源读取。** 来源分解的线性可加性不代表纯模态语义或因果归因。
11. **训练目标与推理 Block 分开。** Relation dual loss、三出口 decision correction、冻结 gate Stage2、pair views 均改变优化方案；source off/near splice/Oracle/frozen probe 只作诊断，不应记成新 Block 提分。
12. **最新 conditional-reread 本地记录只有启动，无正式分数。** free vs span 是受控访问能力对照；不能用 smoke67 tests 或启动 PID 声称性能提升。

## 已核实的关键代表结果（不是跨协议排行榜）

|协议组|方法|种子 / 预算|八率 W-F1|高缺失 W-F1|证据|
|---|---|---|---:|---:|---|
|历史 cfg84|Local-conditioned evidence Gate+L2|66/67/68，100轮|80.286|75.955|osram_local_evidence_gate_20260930/RESULT.md|
|历史 cfg84|History scalar Gate|66/67/68，100轮|79.892|75.239|osram_cfg84_history_input_gate_20260929/RESULT.md|
|历史 cfg84|Post-GRN|66/67/68，100轮|79.521|74.665|osram_cfg84_post_grn_20260929/RESULT.md|
|历史 cfg84|Nested residual|66/67/68，100轮|80.489|75.252|osram_high_score_review_20261009/RESULT.md|
|历史 cfg84|NPS Base/Gap-only|66/67/68，100轮|80.459|75.558|osram_high_score_review_20261009/RESULT.md|
|历史 cfg84|D3-W256 scalar shift filter|66/67/68，100轮|79.953|74.992|osram_readout_path_audit_20261009/RESULT.md|
|历史 cfg84|Nested direct-zero|66，100轮|77.315|71.312|osram_nested_direct_20261008/RESULT.md|
|历史 cfg84|Nested direct-random|66，100轮|80.177|74.800|osram_nested_direct_random_20261008/RESULT.md|
|当前 GCNet mask MOSI|来源对角Attention|66，150轮|78.617|72.976|osram_v3_paired_interfaces_20261022/RESULT.md|
|当前 GCNet mask MOSI|更新Local仅Adapter|66，150轮|79.323|73.551|同上|
|当前 GCNet mask MOSI|更新Local同时Skip|66，150轮|78.871|72.779|同上|
|当前 GCNet mask MOSI|task增量Norm前|66，150轮|78.860|73.156|同上|
|当前 GCNet mask MOSI|task增量Norm后|66，150轮|78.806|72.581|同上|
|当前 GCNet mask MOSI|Flat512+V3width128|66，150轮|79.453|73.519|osram_mosi_v3_capacity_20261022/RESULT.md|
|当前 GCNet mask MOSEI|head-preserving V3|66，60轮|83.293|80.188|osram_v3_connection_screen_20261022/RESULT.md|
|当前 GCNet mask MOSEI|SourceRole Direct|66/67/68，60轮|83.2609|80.0743|osram_source_role_direct_cross_20261020/completed/RESULT.md|

所有分数来源、dataset/seed/fold/epoch 与协议注记同时保存在 `readout_families.json`。没有在上述明确报告中找到的最终值不补猜、不用早期进度值替代。
