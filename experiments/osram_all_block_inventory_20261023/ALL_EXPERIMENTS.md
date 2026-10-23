# 全部 Block 与调参实验总账

INTERNAL DIAGNOSTIC ONLY

盘点时间：2026-10-23T03:15:17.099354+00:00；源码：`aecf6d83481c4eecfd16e89c842af5567cfb190f`。

共覆盖 **484 个实验目录**、**83 条语义家族**、**311 个注册/配置入口**。
另有 4831 条叶级运行证据、285643 条结构化指标证据；它们不是去重后的独立训练次数。

## 使用说明

- `EXPERIMENTS.csv`：每个实验目录一行；不会遗漏只有文档/诊断的目录。
- `RUNS.csv`：能找到运行配置的叶级记录，保留逐 seed/fold/结构/超参数/环境/协议。
- `METRICS.csv`：逐条件/逐率原始指标行；保留JSON指针，不将Test-best和paired拼成同一成绩。
- `BLOCK_CATALOG.csv`：全部注册机制/配置入口；注册不等于已训练。
- `SOURCE_TABLE_ROWS.csv`：所有发现的Markdown表格行，含来源及行号，覆盖无法统一解析的历史汇总。
- `ALL_BLOCKS.html`：可搜索的本地总表，不需要服务器或网络。

## 不可混用的边界

- 新 `gcnet-random-mask-v1`、旧 `conversation-mask-v2`、legacy-fixed及unknown分别标记，不作统一最高分排名。
- Missing未显式记录且没有精确执行源码证明时标unknown，不能按日期或当前默认值猜。
- ACC/UA/W-F1的百分数单位统一；Macro-F1不冒充UA。缺失UA不补造。
- 八率均值只从已保存八率同名指标求等权平均；单率/子集/三种子汇总不会填进叶级八率列。
- 完成状态来自记录声明；只有结果、没有完成证据时写completion_unknown。仅smoke不列为正式完成。
- 相同配置在本地和远端可能有多份证据，不凭相同config自动认定同一次训练。
- 旧结果保留；早期launch/not-trained声明可能已过期，应查看后续完成报告。

## Block/训练家族表

| 家族 | 类型 | 机制/职责 | 调整项 | 实验目录数 | 已有代表结果（不跨协议比较） |
|---|---|---|---|---:|---|
| 早期 Local/历史融合替换 | architecture_or_training_variant | LocalCrossAttentionFusion / LocalCenteredContextFusion；替换旧融合，不等同cfg84后置残差 | local-cross-attn, local-gated, local-centered, modality-tracks | 4 | [] |
| History–Innovation插值读出 | architecture_or_training_variant | 历史与当前映射后scalar K插值；当前注册已移除，只有旧审计引用 | history-innovation（取消的历史归档） | 1 | [] |
| Memory-Shift-Residual / scalar filter | architecture_or_training_variant | Relation128；四证据共享scalar filter；零初始化映射加Flat pre-norm | D1-W128, D2-W128, D2-W256, D3-W128, D3-W256 | 2 | [{"dataset": "MOSI", "epochs": 100, "family_id": "memory_shift", "folds": null, "metrics_percent": {"wf1_high": 76.2, "wf1_mean8": 80.932}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_shift_capacity_20261002/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "D3-W256 scalar shift filter", " |
| Flat后GRN | architecture_or_training_variant | x=Flat hidden，condition=读出消融后的Local/Base/Gap/a；channel gate×value加x | post-GRN开/关 | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "post_grn", "folds": null, "metrics_percent": {"wf1_high": 74.665, "wf1_mean8": 79.521}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_cfg84_post_grn_20260929/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Flat+postGRN", "warning": |
| 总历史scalar Gate | architecture_or_training_variant | alpha=1+0.2tanh，统一缩放Base/三个Gap | one-stage joint | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "history_input_gate", "folds": null, "metrics_percent": {"wf1_high": 75.239, "wf1_mean8": 79.892}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_cfg84_history_input_gate_20260929/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Histo |
| 历史支持条件化Gap Query | architecture_or_training_variant | Query adapter，不是读出Gate | random-missing history-support query | 1 | [] |
| Local-conditioned scalar evidence Gate | architecture_or_training_variant | 每证据单scalar，中心1，L2保持正则 | Base+Gap四gate, Gap-only, L1约束 | 3 | [{"dataset": "MOSI", "epochs": 100, "family_id": "evidence_gate", "folds": null, "metrics_percent": {"wf1_high": 75.955, "wf1_mean8": 80.286}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_local_evidence_gate_20260930/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Local-condition |
| 冻结Flat二阶段Gate | architecture_or_training_variant | 3seeds×8父checkpoint，各训练gate；与一阶段不能混排 | epoch0, oracle-best含epoch0, final100 | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "frozen_evidence_gate", "folds": null, "metrics_percent": {"wf1_high": 75.782, "wf1_mean8": 80.741}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_frozen_evidence_gate_20260930/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Frozen  |
| Feature Gate＋Evidence Softmax | architecture_or_training_variant | 每路内部feature/head过滤再active evidence间竞争；K×softmax | feature＋evidence two-level | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "hierarchical_gate", "folds": null, "metrics_percent": {"wf1_high": 75.267, "wf1_mean8": 80.018}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_hierarchical_evidence_gate_20260930/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Hier |
| 仅8×64 Feature Gate | architecture_or_training_variant | 无evidence-level softmax竞争 | 8×64 feature-only | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "feature_gate", "folds": null, "metrics_percent": {"wf1_high": 75.158, "wf1_mean8": 79.957}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_feature_only_gate_20260930/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Feature-only Gate" |
| Local Skip Gate | architecture_or_training_variant | 只门控完整LocalSkip输出 | scalar/channel配置以run.py为准 | 1 | [] |
| Adapter不含Local | architecture_or_training_variant | 保留原LocalSkip；Adapter只接Base和Gap | memory-only adapter | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "memory_only_adapter", "folds": null, "metrics_percent": {"wf1_high": 74.6, "wf1_mean8": 79.679}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_memory_only_adapter_20261001/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Memory-only |
| Current–History Relation | architecture_or_training_variant | 共享投影+逐证据[q,k,q*k,/q-k/,type]→mean→Flat残差；非Gate | pairwise relation, 同容量pooled control（仅实现未训练）, 0.5base+0.5full任务监督, near-only splice（离线）, relation-off（冻结） | 2 | [{"dataset": "MOSI", "epochs": 100, "family_id": "current_history_relation", "folds": null, "metrics_percent": {"wf1_high": 75.685, "wf1_mean8": 80.576}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_current_history_relation_20261003/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Pairwis |
| Gap增量scalar filter | architecture_or_training_variant | 同轨迹uFull-uBase；LN(uFull+(g-1)increment) | 3seeds | 1 | [{"dataset": "MOSI", "epochs": 100, "family_id": "gap_increment_filter", "folds": null, "metrics_percent": {"wf1_high": 74.699, "wf1_mean8": 79.579}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_gap_increment_filter_20261003/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Gap-inc |
| Evidence-Centered Hierarchical Decision | architecture_or_training_variant | 新Local决策+中心化Base/Gap决策差值；三个出口各1/3任务监督；旧Flat不执行 | Local, Local+Base, Local+Base+Gap | 1 | [] |
| Base-first Gap-correction | architecture_or_training_variant | u=Skip(L)+FB(L,B)，Gap后修正u；不是原Flat保留 | Base-first Gap correction | 1 | [] |
| 多候选/边际读出 | architecture_or_training_variant | 冻结或新训练的边际组合路径；需区分是否单独头 | 原Flat历史recency元数据, counterfactual候选 | 2 | [] |
| 原始Nested GNN及结构消融 | architecture_or_training_variant | Rooted子图GIN，解码修改Adapter输入Local/Base/Gap；Skip保持原Local | old dim64 depth3 groups8, plain-GIN, no-markers, no-head-edges, last-layer, dim32/128, depth1/2, groups1/4, root-aware pooling, Local8, groups1dim512 | 4 | [{"dataset": "MOSI", "epochs": 100, "family_id": "nested", "folds": null, "metrics_percent": {"wf1_high": 75.252, "wf1_mean8": 80.489}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66, 67, 68], "source": "experiments/osram_high_score_review_20261009/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Original Nested residual |
| Nested容量/Flat参数重分配 | architecture_or_training_variant | 扩大图节点宽度或GIN MLP；缩小Flat；必须同时记录Flat/Nested容量 | GIN MLP256/512, Flat256+Nested704, Flat widths384/512/768/1024/1280 | 5 | [{"dataset": "MOSI", "epochs": 100, "family_id": "nested_capacity", "folds": null, "metrics_percent": {"wf1_high": 72.358131, "wf1_mean8": 78.151615}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_nested_equal_budget_20261009/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Flat256+Nested7 |
| Flat容量 / 小Flat+模块 | architecture_or_training_variant | 只改emotion_adapter hidden width；原output_dim1600/Skip不变 | Flat256, Flat256+NPS, Flat256+Nested64 | 2 | [{"dataset": "MOSI", "epochs": 100, "family_id": "small_flat", "folds": null, "metrics_percent": {"wf1_high": 75.89942, "wf1_mean8": 80.362122}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_nested_equal_budget_20261009/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Flat256+Nested64", "w |
| Nested直接替换Adapter证据输入 | architecture_or_training_variant | 去掉输入residual，不删除Flat；zero/random decoder初始化分开 | direct-zero64/64, direct-random64/64, 64/128, 64/256, 64/512, 128/128, 128/256, 128/512, 256/256, 256/512 | 4 | [{"dataset": "MOSI", "epochs": 100, "family_id": "nested_direct", "folds": null, "metrics_percent": {"wf1_high": 71.312, "wf1_mean8": 77.315}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_nested_direct_20261008/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Nested direct zero", "warning |
| Nested训练/LR/epoch/跨数据集 | architecture_or_training_variant | 原图模块不同优化/训练预算，不能当新机制 | 100/150epoch, constant/cosine, 局部LR+warmup+randomDecoder, MOSEI dim32+depth2, MELD d32-l4-g2-m64等 | 11 | [] |
| Nested解码残差Gate/固定系数 | architecture_or_training_variant | 只修改Nested输入校正强度，不能描述成CHM Gate | allGap Gate, GapT-only Gate, fixed residual scale grid | 3 | [] |
| Nested Rooted concat独立读出 | architecture_or_training_variant | 保留rooted表示与Flat表示concat/Probe，不是相同残差架构 | concat, frozenNested-only学习, frozenRepresentationProbe | 3 | [] |
| Nested只读消融/梯度 | frozen_diagnostic_or_archival_review | 无新训练；不是调参方法数 | 单槽关, 全部/组合关, 输入梯度, 训练梯度, 四数据集逐rate汇总 | 6 | [] |
| Neural Production | architecture_or_training_variant | 四规则共享多步选择/更新；原版Base/Gap输入残差，Local改版独立 | Base/Gap-only四规则三步width96, Local+Base/Gap width96, Local+Base/Gap width256 | 2 | [{"dataset": "MOSI", "epochs": 100, "family_id": "nps", "folds": null, "metrics_percent": {"wf1_high": 76.475, "wf1_mean8": 80.981}, "protocol": "historical cfg84 cyclic random missing; per-rate Test-best W-F1; exact sampler version/environment must be joined from run provenance (do not equate with gcnet-random-mask-v1)", "seeds": [66], "source": "experiments/osram_high_score_review_20261009/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Original NeuralProduction BaseGap-o |
| 多机制去输入残差 | architecture_or_training_variant | XCA / NPS /其他输入adapter direct不是全部删Flat | XCA direct, four direct input candidates | 2 | [] |
| B1 R / B2 R+S短窗历史 | architecture_or_training_variant | 历史Local短窗8；B1检索；B2额外训练关系BCE及same/change拆分 | B1输入2435, B2输入2564, B2关系loss0.05, 分支off/S=0.5（诊断） | 4 | [{"dataset": "MELD", "epochs": 50, "family_id": "related_history", "metrics_percent": {"acc_mean8": 61.796, "ua_mean8": 38.502, "wf1_mean8": 58.936}, "protocol": "gcnet-random-mask-v1; cmerc-repro; lr1e-5 wd2e-4; per-rate ACC Test-best same-prediction UA", "seeds": [66], "source": "experiments/osram_meld_related_history_20261018/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "B1"}, {"dataset": "MELD", "epochs": 50, "family_id": "related_history", "metrics_percent": {"acc_me |
| TASR Gap来源轨迹校准 | architecture_or_training_variant | JointSourceTracker精确Value槽来源分解；Gap9路 bounded0.25tanh加原Gap | F0, F1 source-neutral=Gap/3, F2 source-real, adapterLR3e-6/1e-5/3e-5/3e-4, 冻结双视图0.8/0.2, 冻结random-only, joint random-only | 4 | [{"dataset": "MELD", "epochs": 30, "family_id": "tasr", "folds": null, "metrics_percent": {"acc_mean8": 61.307, "persistent6_acc": 57.637, "persistent6_ua": 32.095, "persistent6_wf1": 50.919, "ua_mean8": 38.965, "wf1_mean8": 58.637}, "protocol": "fixed historical miss0.3 Test-oracle backbone; validation-random8 ACC then CE selects adapter; same-checkpoint paired eval; Adam3e-4 wd1e-5; dual task0.8random0.2persistent", "seeds": [66], "source": "experiments/osram_tasr_20261018/RESULT.md", "status" |
| BaseText/GapText候选Router | architecture_or_training_variant | Full/去GapT←T/去Base←T三候选预测，训练概率混合或标量混合；不是TASR | frozenRouter, jointRouter, 主干1e-5/2e-5, IEMOCAP6五折, MOSI新Miss/选点修正 | 10 | [] |
| Base-Gap Partner Source Role | architecture_or_training_variant | 12来源角色共享MLP；原版原读出+0.25tanh权重delta；direct版2sigmoid来源重构 | residual MOSI/MOSEI, direct MOSI3seed, direct MOSEI/MELD/IEMOCAP6, gamma=1冻结 | 5 | [{"dataset": "MOSI", "epochs": 150, "family_id": "coupled_source_role", "folds": null, "metrics_percent": {"wf1_high": 73.1139, "wf1_mean8": 79.0506}, "protocol": "gcnet-random-mask-v1 cyclic0.0–0.7; per-rate Test-best W-F1; s0 environment; MSE; seed66 unless stated", "seeds": [66], "source": "experiments/osram_coupled_source_role_20261020/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Coupled residual", "warning": "Representative archived evidence; not an exhaustive per-r |
| SourceRelation Base/Gap均值包 | architecture_or_training_variant | 每来源Base与masked-meanGap交互→三个128tokens→SA→LocalCA→全局1600残差；原提示明确含mean | MOSI150, MOSEI60 | 1 | [] |
| 12-source-role tokens重做 | architecture_or_training_variant | 不提前meanGap；12个128tokens Attention→全局Flat残差；用户取消未完成MOSEI | MOSI完成150, MOSEI20后取消 | 1 | [{"dataset": "MOSI", "epochs": 150, "family_id": "source_role_tokens_v2", "folds": null, "metrics_percent": {"wf1_high": 73.162256, "wf1_mean8": 78.870661}, "protocol": "gcnet-random-mask-v1 cyclic0.0–0.7; per-rate Test-best W-F1; s0 environment; MSE; seed66 unless stated", "seeds": [66], "source": "experiments/osram_source_role_tokens_20261021/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "12token source-role", "warning": "Representative archived evidence; not an exhausti |
| V3 Hierarchical Source–Role | architecture_or_training_variant | 同源Base-eachGap343pair→同role跨sourceSA→Local roleCA→四零初始化64→512decoders回填Base/Gap | MOSI seed66/67/68, MOSEI, MELD, IEMOCAP4/6五折, classification调度+3seed补充 | 3 | [] |
| V3宽度/Flat宽度协调 | architecture_or_training_variant | 缩小Flat hidden、扩大V3width；容量对照同协议 | Flat512/1024, V3width128/256, Plain匹配control, MOSI f512v128三种子 | 4 | [{"dataset": "MOSI", "epochs": 150, "family_id": "v3_capacity", "folds": null, "metrics_percent": {"wf1_high": 73.519, "wf1_mean8": 79.453}, "protocol": "gcnet-random-mask-v1 cyclic0.0–0.7; per-rate Test-best W-F1; s0 environment; MSE; seed66 unless stated", "seeds": [66], "source": "experiments/osram_mosi_v3_capacity_20261022/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "Flat512+V3width128", "warning": "Representative archived evidence; not an exhaustive per-run record o |
| V3非残差读出替换 | architecture_or_training_variant | V3 decoder输出直接作为Flat Base/Gap输入；不再加原read | MOSI direct, MOSEI direct role | 1 | [] |
| V3三种连接位置 | architecture_or_training_variant | 保head回填/来源槽/Flat隐藏层接口 | head-preserving, source-slots, flat-hidden | 2 | [{"dataset": "MOSI", "epochs": 150, "family_id": "v3_connections", "folds": null, "metrics_percent": {"wf1_high": 73.438, "wf1_mean8": 79.331}, "protocol": "gcnet-random-mask-v1 cyclic0.0–0.7; per-rate Test-best W-F1; s0 environment; MSE; seed66 unless stated", "seeds": [66], "source": "experiments/osram_v3_connection_screen_20261022/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "head-preserving", "warning": "Representative archived evidence; not an exhaustive per-run reco |
| V3六组成对接口 | architecture_or_training_variant | 来源连接拓扑；更新Local与Skip一致性；任务增量Norm前后 | topology-full, topology-diagonal, local-adapter, local-shared, task-pre-norm, task-post-norm | 1 | [{"dataset": "MOSI", "epochs": 150, "family_id": "v3_paired", "folds": null, "metrics_percent": {"wf1_high": 73.713, "wf1_mean8": 79.139}, "protocol": "gcnet-random-mask-v1 cyclic0.0–0.7; per-rate Test-best W-F1; s0 environment; MSE; seed66 unless stated", "seeds": [66], "source": "experiments/osram_v3_paired_interfaces_20261022/RESULT.md", "status": "completed_as_explicitly_reported", "variant": "topology-full", "warning": "Representative archived evidence; not an exhaustive per-run record or c |
| Conditional Memory Reread | architecture_or_training_variant | 首读Local/Base/a生成第二query，再读同一alpha衰减prewriteM；Base校正 | free-query, original-effective-query-span | 1 | [] |
| 历史来源/组合/Query决策冻结诊断 | frozen_diagnostic_or_archival_review | 12路消融+8来源Shapley+BaseTextGapText交互；不是新模块或正式训练 | Local/Base/Gap组合, 12格移除, 3source Shapley, query vs readout frozen probe | 12 | [] |
| 历史读出/高分审查 | frozen_diagnostic_or_archival_review | 已有结果证据整理，不增加训练 | placement review, historical shortlist | 5 | [] |
| MELD B3 MBT瓶颈局部/HPO | architecture_or_training_variant | 同Block2输入adapter；bottleneck数/交换轮数与局部参数搜索 | tokens1..8, rounds1..4, 默认4/3, 局部宽度/学习率参数按trialconfig | 5 | [] |
| MELD B6 Learned Robust PCA局部/HPO | architecture_or_training_variant | unrolled low-rank/sparse evidence decomposition输入校正 | rank, 迭代步数, 阈值初始化, B6独立LR, training lr/wd/dropout | 5 | [] |
| MELD六Block+同协议Flat筛选 | architecture_or_training_variant | 统一MeaningfulInputAdapter路由；不是六种完全新机制 | B1 graph_matching_base_gap_pairs, B2 dual_attention_symbolic_relations, B3 mbt_role_bottleneck, B4 conditional_new_07_neural_production, B5 tokenlearner_fuser_v11, B6 learned_robust_pca_evidence_decomposition | 1 | [] |
| MELD Flat宽度/Local Skip结构对照 | architecture_or_training_variant | 改变Flat adapter容量/去LocalSkip重新训练及冻结证据消融 | adapter512等宽度, LocalSkip on/off, BaseGap readout combinations（诊断） | 6 | [] |
| 双视图 task-only / CoMM-style paired history | training_strategy | View1原cyclic；View2仅额外删已可见模态，非空；共享参数、两次独立零初始Memory，两个task loss。 | history_drop_prob=0.2, rho=0.5, epochs=100, View2非空恢复规则 | 1 | [{"baseline": 81.068, "dataset": "MOSI", "method": 80.386, "metric": "8-rate W-F1", "seed": 66}, {"baseline": 67.639, "dataset": "MOSI", "method": 71.583, "metric": "persistent-six W-F1", "seed": 66}] |
| 双视图 + symmetric InfoNCE | training_objective | 当前availability相同且严格过去不同的anchor；1600→256→128 projector；排除同conversation negatives。 | lambda_con=0.1, temperature=0.1, rho=0.5或0.1, history_drop_prob=0.2 | 2 | [{"dataset": "MOSI", "infonce_wf1": 80.267, "rho": 0.5, "seed": 66, "task_only_wf1": 80.386}, {"dataset": "MOSI", "infonce_wf1": 80.191, "rho": 0.1, "seed": 66, "task_only_wf1": 81.231}] |
| 双视图task loss权重rho扫参及三种子 | training_hpo | (1-rho)*L_view1+rho*L_view2，其他配置保持A不变。 | rho:0.05/0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.45/0.60, rho0.5为既有对照, 复验rho0.05/0.10/0.25×seeds66/67/68 | 1 | [{"flat_wf1": 81.068, "scope": "seed66 rho.1", "wf1": 81.231}, {"flat_wf1": 80.559, "scope": "3seed rho.1", "wf1": 80.26}, {"flat_wf1": 80.559, "scope": "3seed rho.25", "wf1": 80.014}] |
| 双视图task-only迁移MELD/IEMOCAP6 | training_strategy_transfer | rho.5/drop.2；不改变推理架构；IEMOCAP6五折seed66。 | MELD50epochs/lr1e-5/wd1e-5, IEMOCAP6 100epochs/lr1e-3/wd1e-5 | 1 | [{"dataset": "MELD", "delta": [-0.211, 0.552, 0.037], "metric": "Random8 ACC/UA/WF1 delta pp"}, {"dataset": "IEMOCAP6", "delta": [-0.278, -0.171, 0.087], "metric": "Random8 ACC/UA/WF1 delta pp"}, {"dataset": "IEMOCAP6", "delta": [0.805, 0.501, 1.038], "metric": "persistent-six ACC/UA/WF1 delta pp"}] |
| MC-FWC未来写入credit及E0–E4对照 | write_directed_training_objective | 同输入Keep/Omit单次源写入反事实；tau softplus调制future task gradient；E4仅source K/V projector VJP，普通task全模型BPTT不变。 | H=4, tau=1.0, lambda_fwc=0.1, Random/Transition/Persistent轨迹比例, E0 original/E1 MixedERM/E2uniform/E3fullgradient/E4writeronly/shuffled | 1 | [] |
| Complete-View Local-State JEPA | training_objective | 完整视图Local-state target替代原modality目标，保留原emotion任务。 | state target/predictor, state loss weight, EMA, 100epochs/seeds66–70 | 1 | [{"emotion_only": 80.076, "joint": 80.002, "metric": "8-rate W-F1", "state_jepa": 79.89}] |
| WSC / Write-State JEPA | training_objective | 写后状态监督；与Joint/No-JEPA/State-JEPA对照。 | write-state loss, 参数预算匹配, EMA, 100epochs/seeds66–70 | 1 | [{"joint": 80.002, "metric": "8-rate W-F1", "no_jepa": 80.076, "wsc": 79.787}] |
| Future-State JEPA | training_objective | 当前post-write Memory→state query/predictor→下一句full-view EMA Local；邻接合法时刻监督。 | future weight=.1, SmoothL1, EMA, 参数匹配hidden487（当H8/d32/output700） | 1 | [] |
| 监督预训练Frozen Teacher替代EMA目标 | teacher_training | Stage1 complete emotion训练projector，Stage2使用冻结Teacher targets；student仍原任务和JEPA。 | teacher_mode=ema/pretrained-frozen, Stage1选点与成本, source/projector维度 | 1 | [] |
| Missing-Text predictable subspace | teacher_target_representation | Stage1训练R/C/Q的Text32维目标、sentiment/predictability/variance/covariance约束；Stage2冻结R辅助Text-only目标。 | target_dim32, variance_floor1, covariance正则, beta1, target_space | 1 | [] |
| M3 utterance-level跨数据预训练及MOSI-only控制 | pretraining | 三数据集均匀抽样、全部9个有向模态预测路径；MOSI-only按24 vs71updates/epoch匹配MOSI暴露。 | dataset mixing, steps_per_epoch, 100epochs, seeds66–70, reg/cl预测分支 | 2 | [{"joint": 0.1433, "metric": "MOSI validation all-route centered cosine", "mosi_only": 0.1019}] |
| Source-only Pre-OSRAM Completion（旧B2） | input_completion | source-only MMoE先预训；预测缺失latent经CompletedReadFusion回灌read node，但真实observed write路径不变。 | Stage1/Stage2 100+100epochs, read-node fusion, regression/contrastive branches | 1 | [{"b2_wf1": 79.7584, "p0_wf1": 80.0022, "selection": "per-rate Test-oracle"}, {"b2_wf1": 79.0772, "p0_wf1": 79.4407, "selection": "8-rate-mean Test-oracle singleepoch"}] |
| Joint-pretrained冻结completion/双projector回灌与初始化控制 | input_completion | 三数据预训latent predictor冻结；second projector bank预测只进入read node，不写persistent Memory；另有初始化控制。 | single/dual projector, 冻结/联合训练, completed read fusion, pretrained initialization | 4 | [{"baseline": 80.559, "dual_projector": 78.842, "high_baseline": 75.594, "high_method": 73.412, "metric": "8-rate W-F1"}] |
| Regression-only JEPA/MMoE/simple masked-mean目标对照 | training_objective_hpo | 原fullMMoE保留reg-only vs简单masked-mean predictor vs去全部辅助；复用90规格筛选但排除不兼容teacher维度。 | training_objective=joint-reg-only/emotion-only, simple_regression_predictor, teacher mode, JEPAloss权重, cfg84 output1600/KV64/H8 | 5 | [] |
| MOSI情绪任务定义：binary/threeclass/MSE+BCE | task_objective | 改分类输出/标签或原标量MSE上额外BCE sign loss；不是新的融合Block。 | binary/threeclass/regression, dual loss coefficient, neutral过滤与阈值 | 3 | [{"dual_loss": 80.14, "metric": "8-rate W-F1", "regression": 80.08}] |
| 训练缺失分布：Uniform-forced-Text / cyclic去miss0 | mask_distribution_training | 一组r~Uniform(0,1)+Text强缺失.25；另一组只在.1–.7训练仍测.0–.7。 | forced_text_prob=.25, train_missing_rates, r distribution | 2 | [{"flat_wf1": 80.08, "method_wf1": 79.74, "scope": "uniform forcedText"}, {"flat_wf1": 80.445, "method_wf1": 80.038, "scope": "cyclic去0"}] |
| CED Coalition Evidence Decomposition输入块 | input_block | 对当前真实observed非空子集共享encoder；Möbius/Harsanyi一二三阶分解，三个结构槽concat生成256维node。 | P1/P2/P3, shared coalition dropout, node replacement vs任务侧使用边界 | 1 | [{"added_parameters": 398086, "ced": 79.875, "flat": 81.068, "high_ced": 74.512, "high_flat": 76.352, "metric": "8-rate W-F1"}] |
| One-stage Full-History Teacher独立模型 | alternative_history_architecture | 原observed-set encoder；3层causalTransformer历史encoder；Local predictor+g(e,c)-g(e,0)决策校正，final-onlyMSE。 | width512/heads8/FF1792/layers3, 约11M预算, MOSI66 150epochs/3e-4→3e-5 | 1 | [] |
| 早期OSRAM容量/Head/KeyValue/output维度 | architecture_hpo | OSRAM heads、K/V、Local/output尺寸及forward-only/bidirectional容量配置。 | num_heads, key_dim, value_dim, output_dim, bidirectional, forward_slot_reuse | 13 | [] |
| MOSI90规格优化/容量联合筛选 | training_and_capacity_hpo | 有限90规格而非完整Cartesian：optimization、capacity、schedule分组；源码保留后续自动复验入口。 | lr, batch, Adam/AdamW, weight_decay, dropout, projector_dropout, block_lr_multiplier, clip, schedule, epochs, latent/output/heads/KV | 1 | [{"high_wf1": 76.088, "key_value": 32, "output_dim": 700, "record": "partial60spec table cfg01", "wf1": 80.828}] |
| Flat cfg84大容量/2400宽/200轮派生 | flat_capacity_training | 原Flat隐藏或output容量和训练预算改变，非新增机制。 | adapter hidden width, output_dim, epochs, 与Nested容量分配 | 5 | [] |
| 100→150轮完整续训/constant与降低LR | training_budget_schedule | 完整model/optimizer/RNG从epoch100恢复，后50轮LR1e-4；与constant150等单独记录。 | epochs100/150, stage2lr1e-4, constant150, seed65/66/67/68 | 4 | [{"flat_wf1": 80.747, "nested_wf1": 80.562, "scope": "MOSI三种子150旧协议"}, {"flat_acc": 61.312, "flat_ua": 59.632, "nested_acc": 61.48, "nested_ua": 60.156, "scope": "IEMOCAP6五折三种子150"}] |
| MOSI/MOSEI低LR和batch64比例LR | optimizer_batch_hpo | Flat/Nested相同constant1e-4；MOSEI batch32→64且lr1e-3→2e-3。 | batch32/64, lr1e-3/1e-4/2e-3, epochs60/100/150 | 3 | [{"batch64_lr2e3_wf1": 83.787, "flat32_lr1e4_wf1": 83.939, "nested32_lr1e4_wf1": 84.02, "scope": "旧MOSEI66 60epochs"}, {"scope": "新MOSEI Flat1e-4 seeds67/68", "wf1": [83.2445, 83.3268]}] |
| 新GCNet MOSI学习率定版3e-4→3e-5 | optimizer_schedule_hpo | 原Flat seed66比较1e-3/5e-4/3e-4，epoch101降10倍；3e-4补67/68。 | base_lr1e-3/5e-4/3e-4, epoch101gamma.1, 150epochs | 2 | [{"high_lr3e4": 73.4715, "high_original": 72.9079, "lr3e4": 79.2817, "metric": "3seed8rate W-F1", "original1e3": 78.8909}, {"metric": "newFlat66 W-F1", "wf1": 79.2491}] |
| MELD Flat LR/WD优化与锁定baseline | optimizer_hpo | 只改Adam LR与weight_decay；先8配置66，再lr1e-5的WD控制、三种子。 | lr2e-4/1e-4/3e-5/1e-5, wd0/1e-5/1e-4/2e-4/5e-4/1e-3, 50epochs | 4 | [{"acc": 61.786, "scope": "O6 seed66 lr1e-5 wd2e-4", "ua": 38.435, "wf1": 58.918}, {"acc": 61.557, "scope": "current3seed lr1e-5 wd1e-5", "ua": 37.728, "wf1": 58.509}] |
| MELD warmup+cosine及step.3三种子 | schedule_and_memory_hpo | 两轮warmup再cosine至0；保持peakLR.001；比较write_step.6与.3。 | lr_schedule=constant/cosine, warmup_ratio=.05, write_step.3/.6 | 2 | [{"acc": 60.749, "scope": "step.6 3seed ACC-selected", "ua": 36.781, "wf1": 57.279}, {"acc": 60.611, "scope": "step.3 3seed ACC-selected", "ua": 36.474, "wf1": 57.077}] |
| 每率独立训练vs单模型cyclic | training_distribution | 8个模型分别固定train/test rate；不是fixed mask，也不是persistent。 | train_rate_mode=fixed/cyclic, fixed_missing_rate.0–.7, MELD50/MOSI150 | 3 | [{"cyclic_wf1": 79.249, "fixed_wf1": 78.681, "scope": "MOSI66"}, {"fixed_acc": 60.886, "fixed_ua": 37.069, "fixed_wf1": 57.586, "scope": "MELD66"}] |
| MELD写入eta/ridge五配置从头训练 | memory_hpo | 只改变联合写入scalar，CHM方程/KeyQuery/Flat均不变。 | write_step.3/.6/.9, write_ridge1e-4/1e-3/1e-2 | 1 | [{"acc": 60.838, "scope": "W0step.6ridge.001", "ua": 35.633, "wf1": 56.789}, {"acc": 60.948, "scope": "W1step.3", "ua": 35.653, "wf1": 57.054}, {"acc": 60.56, "scope": "W4ridge.01", "ua": 36.717, "wf1": 56.711}] |
| MOSI写入强度与每Head维度五档 | memory_and_capacity_hpo | eta五档和KV五档共享一个baseline，共9独立配置；heads固定8。 | eta=.2/.4/.6/.8/1, K=V=32/48/64/96/128, head48补3seed, head48+step.4 seed66 | 3 | [{"high_wf1": 73.471, "parameters": 13509793, "scope": "3seedhead64", "wf1": 79.282}, {"high_wf1": 73.51, "parameters": 11245857, "scope": "3seedhead48", "wf1": 79.293}] |
| 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 | diagnostic_not_training | 固定模型覆盖eta/alpha/beta或屏蔽head，重放状态/读取/任务；训练后参数提取与关联保留审计。 | frozen alpha .95/.98/.995, frozen beta .05/.5/.95, eta override, leave-head-out, history distance, write interventions | 7 | [{"scope": "MELD trainedalpha means seeds66/67/68", "values": [0.974787, 0.975972, 0.974309]}, {"scope": "MELD trainedbeta means seeds66/67/68", "values": [0.499392, 0.499456, 0.499266]}] |
| 早期write_step1→.6训练及三种query/write参数衍生 | memory_training_and_read_parameters | write_step原默认1到.6从头训练；另有read参数/Gap residual/写入门控runner。 | osram_write_step, write gate, gap_residual_strength, query availability input, history support query conditioning | 4 | [] |
| MELD输入特征切换：ECERC/MAGTKD/MM-Align/MMGCN | dataset_feature_adaptation | 只替换官方预提取特征及必要输入维度，保留原Flat/训练/GCNet missing。 | feature source/utterance coverage, feature input dimensions, pickle SHA256, split/label order | 3 | [{"acc": 60.675, "scope": "historicalECERC66", "ua": 35.436, "wf1": 56.807}, {"acc": 54.631, "scope": "MAGTKD66", "ua": 31.942, "wf1": 51.376}, {"acc": 57.826, "scope": "MMGCN66", "ua": 30.562, "wf1": 53.54}] |
| Missing协议纠正、Test-best UA恢复与环境对账 | protocol_correction_not_block | conversation-mask-v2、gcnet-legacy-fixed-v1与gcnet-random-mask-v1分开；保存ACC最佳同次prediction/mask重算UA。 | missing sampler version, checkpoint_selection_metric, Test-best vs final random retest, s0/adapter/cmerc environment, equalfold/pooled aggregation | 6 | [{"acc": 60.498, "scope": "MELD旧adapter环境GCNet复跑3seed", "ua": null, "wf1": 56.754}, {"acc": 60.693, "scope": "MELD原代码cmerc复跑3seed历史baseline", "ua": 36.255, "wf1": 57.045}] |
| IEMOCAP4/6 Flat新环境五折三种子锁定 | baseline_reproduction | 30fold runs：2数据集×3seed×5fold，复用seed66补67/68。 | 环境一致性, fold1–5, seeds66/67/68, 100epochs | 2 | [{"acc": 79.351, "dataset": "IEMOCAP4", "ua": 79.632, "wf1": 79.233}, {"acc": 61.714, "dataset": "IEMOCAP6", "ua": 60.101, "wf1": 61.362}] |
| ComP因果/非因果复现对照 | external_baseline_not_block | 外部ComP baseline，与Flat/Nested对比；每rate分别训练，非cyclic同一模型。 | causal/noncausal, dataset, seeds, originalrate-specifictraining | 0 | [{"acc": 60.532, "scope": "MELD causal3seed Test-best", "ua": 33.742, "wf1": 56.44}] |
| 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | historical_pre_osram_experiments | 原GCNet+MissingM3阶段的JEPA权重、任务、LR/WD/dropout/hiddenwindow及fixed/cyclic/twostage/frozen/jointcompletion。 | JEPAweight, lr/WD, differentiallr, hidden/window, binary/ordinal, pretrain/freeze/joint, fixed/cyclic schedule | 18 | [] |
| Frozen probes/历史干预/梯度与12格审计 | diagnostic_not_end_to_end_block_training | Frozen heads可训练但不是改原OSRAM；history delete/donor/zero、sourceShapley、gradient冲突等只读冻结诊断。 | probe features, donor matching, history deletion, finite gradient step, fixed masks, threshold仅既定离线分层 | 11 | [] |
| 持续/七种组合/LocalBaseGap推理评价 | evaluation_not_training | 完整模型同checkpoint冻结改最终读出或整段固定observedset；不改参数，不是另一个训练模块。 | Local/Base/Gap on-off, A/T/V/AT/AV/TV/ATV, history_scale alpha, oracle纠错空间, readout interaction | 10 | [] |

## 全部实验目录表

| 实验目录 | 家族 | 数据集 | Seed | Missing | 运行证据 | 指标证据 | 状态 |
|---|---|---|---|---|---:|---:|---|
| 20261013_legacy_mask_smoke | 未归类/见原文 | MELD | 66 | gcnet-legacy-fixed-v1 | 1 | 16 | {"results_present_completion_unknown": 1} |
| asynchronous_state_audit_20260905 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| checkpoint_cleanup_20261011 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| checkpoint_cleanup_approved234_20261011 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| checkpoint_cleanup_priority11_20261011 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| checkpoint_cleanup_round12_20261011 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_all_3seed_300e_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_completion_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_iemocap_3seed_200e_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_iemocap_3seed_200e_cpu4_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_mosei_3seed_100e_cpu4_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_causal_mosi_seed68_300e_cpu4_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_meld | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_meld_causal_50e_3seed_20261012 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_meld_causal_50e_3seed_wf1_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| comp_meld_gcnet_legacy_fixed_50e_3seed_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| dialogue_jepa_20260806 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| dual_roberta_jepa_20260806 | 未归类/见原文 |  |  |  | 0 | 10 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| ecerc_iemocap_features_20261014 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| environment_optuna_20261014 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| frozen-memory-check-GvpYlz | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 241 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| frozen-nested-check-20261009 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| full_history_teacher_20261022 | One-stage Full-History Teacher独立模型 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_all_missing_10trials_20260819 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） |  |  |  | 0 | 160 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_control_osram_protocol_20260906 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"results_present_completion_unknown": 5} |
| gcnet_miss07_10trials_20260819 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） |  |  |  | 0 | 20 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_mosi_no_reconstruction_20260924 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_mosi_paired_patterns_20260925 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_official_5session_iemocap_20260919 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_official_5session_iemocap_20260919_v2 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gcnet_reconstruction_gradient_audit_20260904 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| gpu4_health_20261004 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| iemocap6_fixed_patterns_20260924 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| jepa_all_missing_10trials_20260819 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） |  |  |  | 0 | 160 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| local_cfg84_nojepa_20260919 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_complete_all_rates_20260901 | 未归类/见原文 |  |  |  | 0 | 40 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_complete_context_20260901 | 未归类/见原文 |  |  |  | 0 | 16 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_iemocap_css_20260828 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_iemocap_gcnet_features_20260828 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_mosi_complete_20260828 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_pretrain_mosi_only_20260922 | M3 utterance-level跨数据预训练及MOSI-only控制 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| m3_pretrain_three_datasets_20260921 | M3 utterance-level跨数据预训练及MOSI-only控制 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| meld_block2_methods_cpu_k5p5W2 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| meld_fixed_query_tests | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| meld_fixed_query_validation.d9RDZN | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| miss0_parity_20260819 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| missing_m3_iemocap6_20260828 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_iemocap6_jepa_ablation_20260902 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 10 | 80 | {"complete_declared": 10} |
| missing_m3_iemocap6_no_jepa_20260902 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"complete_declared": 5} |
| missing_m3_iemocap6_stratified_jepa_ablation_20260902 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"complete_declared": 30} |
| missing_m3_iemocap_current_20260831 | 未归类/见原文 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"complete_declared": 30} |
| missing_m3_mosei_current_20260831 | 未归类/见原文 | CMUMOSEI | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"results_present_completion_unknown": 5} |
| missing_m3_mosei_text_anchor_residual_test_oracle_20260904 | 未归类/见原文 | CMUMOSEI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_all_rates_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 5 | 24 | {"promising_but_failed_seed66_gate": 2, "results_present_completion_unknown": 3} |
| missing_m3_mosi_binary_task_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_classification_completion_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_conditioned_readout_20260829 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 96 | 0 | {"complete_declared": 93, "failed": 3} |
| missing_m3_mosi_contrastive_completion_20260903 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| missing_m3_mosi_cyclic_20260830 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_differential_lr_20260831 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66 | unknown | 3 | 24 | {"results_present_completion_unknown": 3} |
| missing_m3_mosi_dual_text_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 3 | 24 | {"results_present_completion_unknown": 3} |
| missing_m3_mosi_fixed_rate_20260830 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 120 | 120 | {"complete_declared": 120} |
| missing_m3_mosi_frozen_completion_20260831 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66 | unknown | 3 | 24 | {"results_present_completion_unknown": 3} |
| missing_m3_mosi_fusion_localization_20260904 | 未归类/见原文 |  |  |  | 0 | 210 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| missing_m3_mosi_graph_branch_ablation_20260829 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"results_present_completion_unknown": 30} |
| missing_m3_mosi_hidden_screen_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 2 | 16 | {"results_present_completion_unknown": 2} |
| missing_m3_mosi_hidden_window_sweep_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68 | unknown | 36 | 288 | {"complete_declared": 36} |
| missing_m3_mosi_jepa_ablation_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_jepa_weight_screen_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66 | unknown | 9 | 72 | {"results_present_completion_unknown": 9} |
| missing_m3_mosi_joint_completion_20260831 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 21 | 168 | {"results_present_completion_unknown": 21} |
| missing_m3_mosi_latent_diagnostic_20260831 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| missing_m3_mosi_latent_oracle_20260903 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| missing_m3_mosi_local_context_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 5 | 24 | {"failed_seed66_gate": 2, "results_present_completion_unknown": 3} |
| missing_m3_mosi_lr_screen_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 33 | 264 | {"results_present_completion_unknown": 33} |
| missing_m3_mosi_modality_track_20260829 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_no_graphconv2_test_oracle_20260905 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_paper_mmoe_20260829 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_pcir_node_20260829 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_postgraph_bilstm_ablation_test_oracle_20260905 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"results_present_completion_unknown": 30} |
| missing_m3_mosi_raw_residual_20260828 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_roberta_lora_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 2 | 16 | {"results_present_completion_unknown": 2} |
| missing_m3_mosi_slot_20260828 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_soft_ordinal_head_20260831 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_target88_search_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 6 | 48 | {"results_present_completion_unknown": 6} |
| missing_m3_mosi_test_oracle_20260831 | 未归类/见原文 | CMUMOSI | 66 | unknown | 5 | 24 | {"complete_diagnostic_only": 2, "results_present_completion_unknown": 3} |
| missing_m3_mosi_text_anchor_residual_20260904 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 27 | 216 | {"results_present_completion_unknown": 27} |
| missing_m3_mosi_text_anchor_residual_stratified_test_oracle_20260905 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| missing_m3_mosi_text_anchor_residual_test_oracle_20260904 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"results_present_completion_unknown": 5} |
| missing_m3_mosi_text_ensemble_20260828 | 未归类/见原文 | CMUMOSI | 67; 68; 69; 70 | unknown | 16 | 368 | {"results_present_completion_unknown": 16} |
| missing_m3_mosi_text_ensemble_confirm_20260828 | 未归类/见原文 | CMUMOSI | 67; 68; 69; 70 | unknown | 8 | 64 | {"results_present_completion_unknown": 8} |
| missing_m3_mosi_timeattn_screen_20260828 | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 8 | {"results_present_completion_unknown": 1} |
| missing_m3_mosi_two_stage_20260831 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66 | unknown | 9 | 48 | {"configuration_or_metadata_only": 3, "results_present_completion_unknown": 6} |
| missing_m3_mosi_weight_decay_screen_20260829 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） | CMUMOSI | 66 | unknown | 9 | 72 | {"results_present_completion_unknown": 9} |
| missing_m3_multidataset_20260828 | 未归类/见原文 | CMUMOSEI; CMUMOSI; IEMOCAPFour | 66; 67; 68; 69; 70 | unknown | 45 | 360 | {"results_present_completion_unknown": 45} |
| missing_m3_raw_sdr_20260830 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 80 | {"complete_declared": 5} |
| missing_m3_sdr_backbone_20260830 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 10 | 160 | {"results_present_completion_unknown": 10} |
| missing_m3_target_ple_20260903 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 20 | 160 | {"results_present_completion_unknown": 20} |
| missing_m3_target_private_ab_20260904 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 10 | 80 | {"results_present_completion_unknown": 10} |
| mmoe_single_expert_mosi_20260907 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| modality_jepa_seed66_20260818 | 早期Missing-M3任务/损失/优化筛选（OSRAM以前） |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| model_screens_scores_20261019T143008Z | 未归类/见原文 | CMUMOSI | 66 | conversation-mask-v2; unknown | 98 | 1568 | {"complete_declared": 98} |
| mosei_nested_gcnet_test_m41WMq | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| mosi_bottleneck_diagnosis_20260907 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| mosi_fixed66_smoke_dTL0p3 | 未归类/见原文 | CMUMOSI | 66 | gcnet-random-mask-v1 | 2 | 2 | {"smoke_complete": 2} |
| mosi_strict_text_only_20260904 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 10 | {"configuration_or_metadata_only": 5} |
| mosi_strict_text_only_test_oracle_20260904 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 15 | {"configuration_or_metadata_only": 5} |
| multimodal_utterance_jepa_20260811 | 未归类/见原文 |  |  |  | 0 | 3 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| nested_sweep_check_wDcrQ8 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| new40_scores_20261019T140245Z | 未归类/见原文 | CMUMOSI | 66 | conversation-mask-v2 | 29 | 464 | {"complete_declared": 29} |
| no_aux_smoke_626zzn4e | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 0 | {"configuration_or_metadata_only": 1} |
| nps-width-check-TVDcc0 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| original_gcnet_iemocap6_stratified_20260902 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 25 | 200 | {"complete_declared": 25} |
| original_gcnet_iemocap6_stratified_cls_only_20260902 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"complete_declared": 5} |
| original_gcnet_reconstruction_gradient_sources_20260904 | 未归类/见原文 | CMUMOSI; IEMOCAPFour | 66 | unknown | 2 | 0 | {"configuration_or_metadata_only": 2} |
| original_missing_sweep_seed66_20260818 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_12_source_evidence_audit_20261019 | 历史来源/组合/Query决策冻结诊断 | IEMOCAPSix; MELD |  | unknown | 4 | 136 | {"diagnostic_complete": 4} |
| osram_ablation_mosi_20260907 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"results_present_completion_unknown": 30} |
| osram_av_text_init_20260919 | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 24 | {"results_present_completion_unknown": 1} |
| osram_b2_20260909 | Source-only Pre-OSRAM Completion（旧B2） | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 200 | {"results_present_completion_unknown": 5} |
| osram_base_first_gap_20261011 | Base-first Gap-correction |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_base_gap_delta_20260921 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_capacity_sweep_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_causal_nojepa_20260910 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 240 | {"results_present_completion_unknown": 15} |
| osram_causal_readout_20260910 | 早期 Local/历史融合替换 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 600 | {"results_present_completion_unknown": 15} |
| osram_ced_20261005 | CED Coalition Evidence Decomposition输入块 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 31 | 231 | {"complete_declared": 10, "configuration_or_metadata_only": 20, "results_present_completion_unknown": 1} |
| osram_cfg84_conversation_mix_20260927 | 未归类/见原文 | CMUMOSI | 66; 67; 68 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_cfg84_fixed_modality_ablation_20260923 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_history_innovation_20260928 | History–Innovation插值读出 | CMUMOSI | 66; 67 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_cfg84_history_input_gate_20260929 | 总历史scalar Gate | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 8 | 128 | {"results_present_completion_unknown": 8} |
| osram_cfg84_history_query_20260927 | 未归类/见原文 | CMUMOSI | 66; 67; 68 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_cfg84_history_query_random_20260928 | 历史支持条件化Gap Query; 早期write_step1→.6训练及三种query/write参数衍生 | CMUMOSI | 66; 67; 68 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_cfg84_history_query_random_base_20260928 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_history_scale_20260929 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_history_scale_fine_20260929 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_memory_shift_residual_20260928 | Memory-Shift-Residual / scalar filter | CMUMOSI | 66; 67; 68 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_cfg84_no_local_20260929 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_original_base_20260928 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_persistent_gap_ablation_20260925 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cfg84_post_grn_20260929 | Flat后GRN | CMUMOSI | 66; 67; 68 | conversation-mask-v2; unknown | 7 | 112 | {"results_present_completion_unknown": 7} |
| osram_complete150_20261011 | 100→150轮完整续训/constant与降低LR |  |  |  | 0 | 1500 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_complete_20260906 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 25 | 200 | {"results_present_completion_unknown": 25} |
| osram_complete_state_20260910 | Complete-View Local-State JEPA | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 240 | {"results_present_completion_unknown": 15} |
| osram_conditional_reread_20261023 | Conditional Memory Reread | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1 | 8 | 320 | {"running": 4, "smoke_complete": 4} |
| osram_context_audit_20261003 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_core20_20261005 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_core20_top3_3seed_20261005 | 未归类/见原文 | CMUMOSI | 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_counterfactual_readouts_20261011 | 多候选/边际读出 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_coupled_source_role_20261020 | Base-Gap Partner Source Role | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1 | 4 | 908 | {"complete_declared": 2, "smoke_complete": 2} |
| osram_cross_substitution_20260910 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_cross_substitution_move_swap_20260910 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_current_history_relation_20261003 | Current–History Relation | CMUMOSI | 66 | conversation-mask-v2 | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_current_pattern_history_audit_20261022 | Frozen probes/历史干预/梯度与12格审计 | CMUMOSI | 66 | gcnet-random-mask-v1 | 3 | 0 | {"complete_declared": 2, "failed": 1} |
| osram_decision_correction_20261003 | Evidence-Centered Hierarchical Decision | CMUMOSI | 66; 67; 68 | conversation-mask-v2; unknown | 9 | 288 | {"results_present_completion_unknown": 9} |
| osram_dev_seed66 | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 8 | {"results_present_completion_unknown": 1} |
| osram_direct_input4_20261010 | 多机制去输入残差 | CMUMOSI | 66 | unknown | 8 | 128 | {"results_present_completion_unknown": 8} |
| osram_directional_20260910 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_dual_projector_memory_write_20260927 | 未归类/见原文 | CMUMOSI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_ecerc_features_20261012 | MELD输入特征切换：ECERC/MAGTKD/MM-Align/MMGCN |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_ecerc_meld | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_ecerc_meld_flat | 未归类/见原文 | MELD | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_ecerc_meld_retry1 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_ecerc_meld_retry2 | 未归类/见原文 | MELD | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_emotion_only_mosi_20260907 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_evidence_gate_l1_20260930 | Local-conditioned scalar evidence Gate | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 8 | 128 | {"results_present_completion_unknown": 8} |
| osram_feature_only_gate_20260930 | 仅8×64 Feature Gate | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_flat_base_gap_interaction_20261011 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_flat_history_recency_20261011 | 多候选/边际读出 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_flat_nested_seed65_150_20261010 | Nested训练/LR/epoch/跨数据集; 100→150轮完整续训/constant与降低LR | CMUMOSI | 65 | unknown | 4 | 88 | {"results_present_completion_unknown": 4} |
| osram_forward_only_iemocap_20260908 | 早期OSRAM容量/Head/KeyValue/output维度 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"results_present_completion_unknown": 30} |
| osram_forward_only_mosi_20260908 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_forward_slot_reuse_20260909 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI; IEMOCAPFour; IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 35 | 280 | {"results_present_completion_unknown": 35} |
| osram_frozen_evidence_gate_20260930 | 冻结Flat二阶段Gate | CMUMOSI | 66; 67; 68 | unknown | 26 | 294 | {"configuration_or_metadata_only": 26} |
| osram_frozen_memory_audit_20261009 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 1328 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_frozen_nested_20261009 | Nested Rooted concat独立读出 |  |  |  | 0 | 24 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_frozen_nested_20261009_retry | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 24 | {"configuration_or_metadata_only": 1} |
| osram_frozen_residual_20261009 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 1728 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_future_state_20260914 | Future-State JEPA | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_gap_increment_audit_20261003 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_gap_increment_filter_20261003 | Gap增量scalar filter | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_gap_only_evidence_gate_20261005 | Local-conditioned scalar evidence Gate | CMUMOSI | 66 | conversation-mask-v2 | 2 | 24 | {"results_present_completion_unknown": 2} |
| osram_gap_query_swap_20260920 | 未归类/见原文 |  |  |  | 0 | 72 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_gcnet_random_mask_20261013 | Missing协议纠正、Test-best UA恢复与环境对账 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_gcnet_random_mask_iemocap_meld_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_gcnet_random_mask_iemocap_meld_20261013_r1 | 未归类/见原文 | IEMOCAPFour; IEMOCAPSix; MELD | 66; 67; 68 | gcnet-random-mask-v1 | 36 | 576 | {"results_present_completion_unknown": 36} |
| osram_gcnet_random_mask_iemocap_nested_20261013 | 未归类/见原文 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | gcnet-random-mask-v1 | 30 | 480 | {"results_present_completion_unknown": 30} |
| osram_heads8_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 11 | 88 | {"results_present_completion_unknown": 11} |
| osram_heads8_20260906_retry | 未归类/见原文 | CMUMOSI | 67; 68; 69; 70 | unknown | 4 | 32 | {"results_present_completion_unknown": 4} |
| osram_heads8_out700_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_heads8_v64_out700_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 14 | 112 | {"results_present_completion_unknown": 14} |
| osram_heads8_v64_out700_20260906_retry | 未归类/见原文 | CMUMOSI | 69 | unknown | 1 | 8 | {"results_present_completion_unknown": 1} |
| osram_hierarchical_evidence_gate_20260930 | Feature Gate＋Evidence Softmax | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_hierarchical_gate_verification_20260930 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_high_score_review_20261009 | 历史读出/高分审查 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_history_content_intervention_20261014 | Frozen probes/历史干预/梯度与12格审计 | IEMOCAPFour; IEMOCAPSix; MELD | 66; 67; 68 | conversation-mask-v2 | 38 | 831 | {"complete_declared": 35, "running": 3} |
| osram_history_drift_20261002 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_history_effect_stratification_20261013 | Frozen probes/历史干预/梯度与12格审计 | IEMOCAPFour; IEMOCAPSix; MELD | 66; 67; 68 | conversation-mask-v2 | 34 | 1011 | {"complete_declared": 34} |
| osram_history_text_context_20261010 | 未归类/见原文 |  |  |  | 0 | 952 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_history_training_controls_20261014 | 未归类/见原文 | IEMOCAPFour | 66 | conversation-mask-v2 | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_iemocap4_20260906 | 未归类/见原文 | IEMOCAPFour | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_iemocap6_20260906 | 未归类/见原文 | IEMOCAPSix | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_iemocap6_gcnet_testbest_20261020 | BaseText/GapText候选Router; Missing协议纠正、Test-best UA恢复与环境对账 | IEMOCAPSix | 66; 67; 68 | gcnet-random-mask-v1 | 31 | 12908 | {"complete_declared": 30, "smoke_complete": 1} |
| osram_iemocap6_router_fivefold_20261019 | BaseText/GapText候选Router | IEMOCAPSix | 66; 67; 68 | gcnet-random-mask-v1 | 25 | 7151 | {"complete_declared": 24, "smoke_complete": 1} |
| osram_iemocap_b6_source_785a012 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_iemocap_b6_transfer_20261015 | MELD B6 Learned Robust PCA局部/HPO | IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | gcnet-random-mask-v1 | 90 | 1080 | {"complete_declared": 90} |
| osram_iemocap_b6_transfer_20261015_preflight | 未归类/见原文 | IEMOCAPSix | 66 | gcnet-random-mask-v1 | 2 | 3 | {"running": 1, "smoke_complete": 1} |
| osram_iemocap_base_text_gap_text_20261019 | 历史来源/组合/Query决策冻结诊断 | IEMOCAPSix |  | unknown | 2 | 540 | {"diagnostic_complete": 2} |
| osram_iemocap_cfg84_nojepa_20260919 | 未归类/见原文 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | unknown | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_iemocap_cfg84_nojepa_5session_20260919 | 未归类/见原文 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | unknown | 34 | 480 | {"complete_declared": 4, "results_present_completion_unknown": 30} |
| osram_iemocap_flat_3seed_cmerc_20261016 | IEMOCAP4/6 Flat新环境五折三种子锁定 | IEMOCAPFour; IEMOCAPSix | 67; 68 | gcnet-random-mask-v1 | 60 | 1280 | {"complete_declared": 60} |
| osram_iemocap_flat_seed66_restore_20261016 | IEMOCAP4/6 Flat新环境五折三种子锁定 | IEMOCAPFour; IEMOCAPSix | 66 | gcnet-random-mask-v1 | 30 | 520 | {"complete_declared": 30} |
| osram_iemocap_flat_source_combinations_20261013 | 历史来源/组合/Query决策冻结诊断; 持续/七种组合/LocalBaseGap推理评价 | IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | unknown | 32 | 2667 | {"complete_declared": 31, "failed": 1} |
| osram_iemocap_information_availability_20261018 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 106080 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_joint_dual_projector_completion_20260922 | Joint-pretrained冻结completion/双projector回灌与初始化控制 | CMUMOSI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_joint_dual_projector_completion_20260922_smoke | 未归类/见原文 | CMUMOSI | 66 | unknown | 1 | 0 | {"configuration_or_metadata_only": 1} |
| osram_joint_frozen_reinjection_20260922 | Joint-pretrained冻结completion/双projector回灌与初始化控制 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_joint_pretrained_fixed_modality_ablation_20260923 | Joint-pretrained冻结completion/双projector回灌与初始化控制 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_joint_pretrained_init_control_20260922 | Joint-pretrained冻结completion/双projector回灌与初始化控制 | CMUMOSI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_kv64_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 14 | 112 | {"results_present_completion_unknown": 14} |
| osram_kv64_20260906_retry | 未归类/见原文 | CMUMOSI | 70 | unknown | 1 | 8 | {"results_present_completion_unknown": 1} |
| osram_kv64_out700_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_kv_task_utility_20261022 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 | CMUMOSI | 66 | gcnet-random-mask-v1 | 4 | 0 | {"complete_declared": 4} |
| osram_local_cross_attn_20260910 | 早期 Local/历史融合替换 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 240 | {"results_present_completion_unknown": 15} |
| osram_local_evidence_gate_20260930 | Local-conditioned scalar evidence Gate | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_local_evidence_gate_verification_20260930 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_local_gated_20260910 | 早期 Local/历史融合替换 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_local_skip_gate_20261001 | Local Skip Gate | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_lr_batch_screen_20261011 | MOSI/MOSEI低LR和batch64比例LR | CMUMOSEI; CMUMOSI | 66 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_lr_screen_mosi_20260907 | MOSI/MOSEI低LR和batch64比例LR | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 240 | {"results_present_completion_unknown": 30} |
| osram_magnitude_20260910 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_masked_mean_reg_20260919 | Regression-only JEPA/MMoE/simple masked-mean目标对照 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_mc_fwc_20261011 | MC-FWC未来写入credit及E0–E4对照 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meaningful20_20261003 | 未归类/见原文 | CMUMOSI | 66 | unknown | 20 | 320 | {"results_present_completion_unknown": 20} |
| osram_meaningful20_round2_20261004 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_medium_flat_nested_20261009 | Nested容量/Flat参数重分配; Flat cfg84大容量/2400宽/200轮派生 | CMUMOSI | 66 | unknown | 10 | 160 | {"results_present_completion_unknown": 10} |
| osram_meld_alternate_features_20261017 | MELD输入特征切换：ECERC/MAGTKD/MM-Align/MMGCN | MELD | 66 | gcnet-random-mask-v1 | 5 | 31 | {"complete_declared": 3, "failed": 1, "smoke_complete": 1} |
| osram_meld_b3_capacity_hpo_20261016 | MELD B3 MBT瓶颈局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 17 | 515 | {"complete_declared": 16, "smoke_complete": 1} |
| osram_meld_b3_default43_3seed_20261016 | MELD B3 MBT瓶颈局部/HPO | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 12 | 156 | {"complete_declared": 12} |
| osram_meld_b3_default43_controller_20261016 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_b3_fixed43_controller_52886f7 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_meld_b3_fixed43_hpo_20261016 | MELD B3 MBT瓶颈局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 16 | 512 | {"complete_declared": 16} |
| osram_meld_b3_gpu3_controller_005616d | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_meld_b3_gpu3_extension_20261016 | MELD B3 MBT瓶颈局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 12 | 392 | {"complete_declared": 12} |
| osram_meld_b3_structure_hpo_20261016 | MELD B3 MBT瓶颈局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 81 | 2563 | {"complete_declared": 80, "smoke_complete": 1} |
| osram_meld_b6_structure_hpo_20261015 | MELD B6 Learned Robust PCA局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 119 | 4303 | {"complete_declared": 118, "smoke_complete": 1} |
| osram_meld_b6_threshold_hpo_20261015 | MELD B6 Learned Robust PCA局部/HPO | MELD | 66 | gcnet-random-mask-v1 | 81 | 2563 | {"complete_declared": 80, "smoke_complete": 1} |
| osram_meld_b6_top2_matched_flat_3seed_20261015 | MELD B6 Learned Robust PCA局部/HPO | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 21 | 566 | {"complete_declared": 15, "failed": 6} |
| osram_meld_base_text_gap_text_20261019 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 540 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_block2_auto_20261014 | MELD六Block+同协议Flat筛选 | MELD | 66 | gcnet-random-mask-v1 | 8 | 216 | {"complete_declared": 7, "failed": 1} |
| osram_meld_block2_auto_20261014_v2 | 未归类/见原文 | MELD | 66 | gcnet-random-mask-v1 | 203 | 5247 | {"complete_declared": 196, "smoke_complete": 7} |
| osram_meld_block2_dev_20261014 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_meld_block2_dev_all_smokes_20261014 | 未归类/见原文 | MELD | 66 | gcnet-random-mask-v1 | 7 | 21 | {"smoke_complete": 7} |
| osram_meld_block2_dev_flat_smoke_20261014 | 未归类/见原文 | MELD | 66 | gcnet-random-mask-v1 | 1 | 0 | {"failed": 1} |
| osram_meld_block2_dev_flat_smoke_v2_20261014 | 未归类/见原文 | MELD | 66 | gcnet-random-mask-v1 | 1 | 3 | {"smoke_complete": 1} |
| osram_meld_block2_topacc_3seed_20261015 | MELD B6 Learned Robust PCA局部/HPO | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 39 | 498 | {"complete_declared": 33, "failed": 6} |
| osram_meld_cosine_20261017 | MELD warmup+cosine及step.3三种子 | CMUMOSI; MELD | 66; 67; 68; 69; 70 | gcnet-random-mask-v1; unknown | 36 | 353 | {"complete_declared": 16, "configuration_or_metadata_only": 20} |
| osram_meld_cosine_step03_3seed_20261017 | MELD warmup+cosine及step.3三种子 | CMUMOSI; MELD | 66; 67; 68; 69; 70 | gcnet-random-mask-v1; unknown | 57 | 795 | {"complete_declared": 37, "configuration_or_metadata_only": 20} |
| osram_meld_fixed_lr_wd_20261018 | MELD Flat LR/WD优化与锁定baseline | MELD | 67; 68 | gcnet-random-mask-v1 | 44 | 544 | {"complete_declared": 44} |
| osram_meld_fixed_query_task_20261014 | 未归类/见原文 |  |  |  | 0 | 17384 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_fixed_rates_20261018 | 每率独立训练vs单模型cyclic | MELD | 66 | gcnet-random-mask-v1 | 24 | 60 | {"complete_declared": 24} |
| osram_meld_flat_history_scale_20261013 | 未归类/见原文 |  |  |  | 0 | 240 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_flat_source_combinations_20261013 | 历史来源/组合/Query决策冻结诊断; 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 336 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_flat_width_20261014 | MELD Flat宽度/Local Skip结构对照 | MELD | 66 | gcnet-random-mask-v1 | 3 | 116 | {"results_present_completion_unknown": 3} |
| osram_meld_flat_width_20261014_multiseed | 未归类/见原文 | MELD | 67; 68 | gcnet-random-mask-v1 | 2 | 84 | {"results_present_completion_unknown": 2} |
| osram_meld_gcnet_legacy_fixed_50e_3seed_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_gcnet_random_no_local_skip_20261014 | MELD Flat宽度/Local Skip结构对照 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_gcnet_random_no_skip_20261014 | 未归类/见原文 | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_meld_gcnet_random_no_skip_evidence_20261014 | MELD Flat宽度/Local Skip结构对照 |  |  |  | 0 | 100 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_gcnet_random_no_skip_evidence_determinism_20261014 | 未归类/见原文 |  |  |  | 0 | 64 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_information_availability_20261014 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 9776 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_legacy_mask_20261013 | Missing协议纠正、Test-best UA恢复与环境对账 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_local_skip_ablation_20261014 | MELD Flat宽度/Local Skip结构对照 |  |  |  | 0 | 29 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_memory_sensitivity_20261017 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 1645 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_meld_mmgcn_features_20261018 | MELD输入特征切换：ECERC/MAGTKD/MM-Align/MMGCN | MELD | 66 | gcnet-random-mask-v1 | 4 | 31 | {"complete_declared": 3, "smoke_complete": 1} |
| osram_meld_nested_local14_50e_20261012 | 未归类/见原文 | MELD | 66 | unknown | 14 | 224 | {"results_present_completion_unknown": 14} |
| osram_meld_nested_screen_20261012 | Nested训练/LR/epoch/跨数据集 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_nested_screen_50e_20261012 | 未归类/见原文 | MELD | 66; 67; 68 | unknown | 70 | 1120 | {"results_present_completion_unknown": 70} |
| osram_meld_nested_top7_3seed_50e_20261012 | 未归类/见原文 | MELD | 67; 68 | unknown | 14 | 224 | {"results_present_completion_unknown": 14} |
| osram_meld_no_skip_evidence_ablation_20261014 | MELD Flat宽度/Local Skip结构对照 |  |  |  | 0 | 40 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_optimizer_screen_20261018 | MELD Flat LR/WD优化与锁定baseline | MELD | 66 | gcnet-random-mask-v1 | 42 | 486 | {"complete_declared": 42} |
| osram_meld_persistent_unimodal_20261018 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 36 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_query_decision_20261014 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 3636 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_query_decision_20261014_smoke | 未归类/见原文 |  |  |  | 0 | 7 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_query_decision_20261014_smoke_v2 | 未归类/见原文 |  |  |  | 0 | 71 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_query_decision_design_20261014 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_related_history_20261018 | B1 R / B2 R+S短窗历史 | MELD | 66 | gcnet-random-mask-v1 | 16 | 172 | {"complete_declared": 12, "failed": 2, "smoke-complete": 2} |
| osram_meld_related_history_diagnostic_20261018 | B1 R / B2 R+S短窗历史 |  |  |  | 0 | 210 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_related_history_learning_audit_20261018 | B1 R / B2 R+S短窗历史 |  |  |  | 0 | 40 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_retrained_no_local_skip_20261014 | MELD Flat宽度/Local Skip结构对照 | MELD | 66 | gcnet-legacy-fixed-v1 | 1 | 22 | {"results_present_completion_unknown": 1} |
| osram_meld_small_lr_3seed_20261018 | MELD Flat LR/WD优化与锁定baseline | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 12 | 144 | {"complete_declared": 12} |
| osram_meld_small_lr_readout_20261018 | 未归类/见原文 | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 6 | 390 | {"complete_declared": 6} |
| osram_meld_testbest_ua_20261014 | Missing协议纠正、Test-best UA恢复与环境对账 | MELD | 67; 68 | gcnet-random-mask-v1 | 2 | 212 | {"results_present_completion_unknown": 2} |
| osram_meld_visual_zero_audit_20261017 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_meld_wd_seed66_20261018 | MELD Flat LR/WD优化与锁定baseline | MELD | 66; 67; 68 | gcnet-random-mask-v1 | 27 | 322 | {"complete_declared": 27} |
| osram_meld_write_retention_20261014 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 | CMUMOSI; MELD | 66; 67; 68; 69; 70 | gcnet-random-mask-v1; unknown | 61 | 446 | {"complete_declared": 20, "configuration_or_metadata_only": 40, "results_present_completion_unknown": 1} |
| osram_meld_write_tuning_20261017 | MELD写入eta/ridge五配置从头训练 | CMUMOSI; MELD | 66; 67; 68; 69; 70 | gcnet-random-mask-v1; unknown | 55 | 505 | {"complete_declared": 35, "configuration_or_metadata_only": 20} |
| osram_memory_only_adapter_20261001 | Adapter不含Local | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 7 | 112 | {"results_present_completion_unknown": 7} |
| osram_memory_replay_20260920 | 未归类/见原文 |  |  |  | 0 | 56 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_module_connection_audit_20261022 | 历史读出/高分审查 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosei_20260906 | 未归类/见原文 | CMUMOSEI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_mosei_cfg84_nojepa_20260919 | 未归类/见原文 | CMUMOSEI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_mosei_dim32_depth2_followup_20261012 | Nested训练/LR/epoch/跨数据集 | CMUMOSEI | 66; 67; 68 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_mosei_direct_role_readout_20261021 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 256 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosei_gcnet_flat_lr1e4_20261021 | MOSI/MOSEI低LR和batch64比例LR | CMUMOSEI | 67; 68 | gcnet-random-mask-v1 | 2 | 349 | {"complete_declared": 2} |
| osram_mosei_gcnet_router_20261020 | BaseText/GapText候选Router | CMUMOSEI | 66 | gcnet-random-mask-v1 | 3 | 445 | {"complete_declared": 2, "smoke_complete": 1} |
| osram_mosei_nested_gcnet_3seed_20261021 | Nested训练/LR/epoch/跨数据集 | CMUMOSEI | 66; 67; 68 | gcnet-random-mask-v1 | 6 | 590 | {"complete_declared": 5, "smoke_complete": 1} |
| osram_mosei_nested_tuning_20261011 | Nested训练/LR/epoch/跨数据集 | CMUMOSEI | 66 | unknown | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_mosei_source_role_switch_20261021 | Base-Gap Partner Source Role |  |  |  | 0 | 196 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosei_v3_capacity_20261022 | V3宽度/Flat宽度协调 | CMUMOSEI | 66 | gcnet-random-mask-v1 | 4 | 340 | {"complete_declared": 2, "smoke_complete": 2} |
| osram_mosei_v3_grid_20261022 | V3宽度/Flat宽度协调 | CMUMOSEI | 66 | gcnet-random-mask-v1 | 8 | 590 | {"complete_declared": 4, "smoke_complete": 4} |
| osram_mosi_base_gap_delta_20260921 | 未归类/见原文 | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_mosi_binary_20260918 | MOSI情绪任务定义：binary/threeclass/MSE+BCE | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 1580 | {"results_present_completion_unknown": 5} |
| osram_mosi_binary_20260918_aborted_gpu23_114039 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_cfg84_cfg23_20260919 | 未归类/见原文 | CMUMOSI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_mosi_cfg84_nojepa_fixed_rate_20260920 | 每率独立训练vs单模型cyclic | CMUMOSI | 66 | unknown | 8 | 8 | {"configuration_or_metadata_only": 8} |
| osram_mosi_dual_loss_20260918 | MOSI情绪任务定义：binary/threeclass/MSE+BCE | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 1580 | {"results_present_completion_unknown": 5} |
| osram_mosi_fixed_rate_seed66_20261021 | 每率独立训练vs单模型cyclic | CMUMOSI | 66 | gcnet-random-mask-v1 | 16 | 32 | {"complete_declared": 16} |
| osram_mosi_gap_query_swap_20260920 | 未归类/见原文 |  |  |  | 0 | 45 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_gcnet_router_3seed_20261019 | BaseText/GapText候选Router | CMUMOSI | 66; 67; 68 | gcnet-random-mask-v1 | 7 | 630 | {"complete_declared": 6, "smoke_complete": 1} |
| osram_mosi_gcnet_testbest_20261020 | Missing协议纠正、Test-best UA恢复与环境对账 | CMUMOSI | 66; 67; 68 | gcnet-random-mask-v1; unknown | 8 | 1178 | {"complete_declared": 6, "smoke_complete": 1, "smoke_failed": 1} |
| osram_mosi_head48_3seed_20261022 | MOSI写入强度与每Head维度五档 | CMUMOSI | 67; 68 | gcnet-random-mask-v1 | 2 | 288 | {"complete_declared": 2} |
| osram_mosi_head48_step04_20261022 | MOSI写入强度与每Head维度五档 | CMUMOSI | 66 | gcnet-random-mask-v1 | 1 | 130 | {"complete_declared": 1} |
| osram_mosi_hparam_sweep_20260918 | MOSI90规格优化/容量联合筛选 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_hparam_sweep_20260918_parallel | 未归类/见原文 | CMUMOSI | 66 | unknown | 90 | 1440 | {"results_present_completion_unknown": 90} |
| osram_mosi_hparam_verify_20260918 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 25 | 400 | {"results_present_completion_unknown": 25} |
| osram_mosi_hparam_verify_selected_20260918 | 未归类/见原文 | CMUMOSI | 67; 68 | unknown | 12 | 192 | {"results_present_completion_unknown": 12} |
| osram_mosi_lr_curve_audit_20261021 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_meld_history_length_audit_20261013 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_memory_gap_ablation_20260920 | 持续/七种组合/LocalBaseGap推理评价 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 20 | 800 | {"results_present_completion_unknown": 20} |
| osram_mosi_memory_grid_20261022 | MOSI写入强度与每Head维度五档 | CMUMOSI | 66 | gcnet-random-mask-v1 | 18 | 1550 | {"complete_declared": 9, "smoke_complete": 9} |
| osram_mosi_mid_lr_20261021 | 新GCNet MOSI学习率定版3e-4→3e-5 | CMUMOSI | 66 | gcnet-random-mask-v1; unknown | 4 | 455 | {"complete_declared": 2, "failed": 2} |
| osram_mosi_mid_lr_3seed_20261021 | 新GCNet MOSI学习率定版3e-4→3e-5 | CMUMOSI | 67; 68 | gcnet-random-mask-v1 | 2 | 206 | {"complete_declared": 2} |
| osram_mosi_modality_tracks_20260918 | 早期 Local/历史融合替换 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 1580 | {"results_present_completion_unknown": 5} |
| osram_mosi_pattern_20260909 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_recent_history_20261018 | B1 R / B2 R+S短窗历史 | CMUMOSI | 66 | unknown | 3 | 80 | {"results_present_completion_unknown": 3} |
| osram_mosi_retention_intervention_20260920 | 未归类/见原文 |  |  |  | 0 | 24 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_retention_replay_20260920 | 未归类/见原文 |  |  |  | 0 | 5 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_router_lr1e5_20261020 | BaseText/GapText候选Router | CMUMOSI | 66 | gcnet-random-mask-v1 | 3 | 445 | {"complete_declared": 2, "smoke_complete": 1} |
| osram_mosi_router_lr2e5_20261020 | BaseText/GapText候选Router | CMUMOSI | 66 | gcnet-random-mask-v1 | 3 | 445 | {"complete_declared": 2, "smoke_complete": 1} |
| osram_mosi_source_role_direct_20261020 | Base-Gap Partner Source Role | CMUMOSI | 66 | gcnet-random-mask-v1 | 2 | 361 | {"complete_declared": 1, "smoke_complete": 1} |
| osram_mosi_source_role_direct_3seed_20261020 | Base-Gap Partner Source Role | CMUMOSI | 66; 67; 68 | gcnet-random-mask-v1 | 3 | 815 | {"complete_declared": 2, "smoke_complete": 1} |
| osram_mosi_source_role_neutral_audit_20261020 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 1314 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_mosi_threeclass_20260918 | MOSI情绪任务定义：binary/threeclass/MSE+BCE | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 1580 | {"results_present_completion_unknown": 5} |
| osram_mosi_v3_capacity_20261022 | V3宽度/Flat宽度协调 | CMUMOSI | 66 | gcnet-random-mask-v1 | 12 | 900 | {"complete_declared": 6, "smoke_complete": 6} |
| osram_mosi_v3_capacity_3seed_20261022 | V3宽度/Flat宽度协调 | CMUMOSI | 67; 68 | gcnet-random-mask-v1 | 4 | 392 | {"complete_declared": 2, "smoke_complete": 2} |
| osram_mosi_v3_direct_20261022 | V3非残差读出替换 | CMUMOSI | 66 | gcnet-random-mask-v1; unknown | 3 | 200 | {"complete_declared": 1, "smoke_complete": 1, "smoke_failed": 1} |
| osram_mosi_write_gate_gap_strength_20260920 | 早期write_step1→.6训练及三种query/write参数衍生 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 600 | {"results_present_completion_unknown": 15} |
| osram_nested_concat_20261011 | Nested Rooted concat独立读出 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_cosine_20261010 | Nested训练/LR/epoch/跨数据集 | CMUMOSI | 66 | unknown | 2 | 48 | {"results_present_completion_unknown": 2} |
| osram_nested_cross_dataset_20261006 | Nested训练/LR/epoch/跨数据集 | CMUMOSEI; IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | unknown | 66 | 2888 | {"results_present_completion_unknown": 66} |
| osram_nested_delta_off_missing_text_20261010 | Nested只读消融/梯度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| osram_nested_delta_subsets_20261011 | Nested只读消融/梯度 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_diagnostics_20261007 | Nested训练/LR/epoch/跨数据集 | CMUMOSEI; CMUMOSI; IEMOCAPFour; IEMOCAPSix | 66; 67; 68 | unknown | 36 | 3240 | {"complete_declared": 36} |
| osram_nested_direct_20261008 | Nested直接替换Adapter证据输入 | CMUMOSI | 66 | unknown | 1 | 46 | {"results_present_completion_unknown": 1} |
| osram_nested_direct_capacity_20261011 | Nested直接替换Adapter证据输入 | CMUMOSI | 66 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_nested_direct_capacity_more_20261011 | Nested直接替换Adapter证据输入 | CMUMOSI | 66 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_nested_direct_random_20261008 | Nested直接替换Adapter证据输入 | CMUMOSI | 66 | unknown | 1 | 26 | {"results_present_completion_unknown": 1} |
| osram_nested_equal_budget_20261009 | Nested容量/Flat参数重分配 | CMUMOSI | 66 | unknown | 1 | 24 | {"results_present_completion_unknown": 1} |
| osram_nested_extend150_20261010 | Nested训练/LR/epoch/跨数据集; 100→150轮完整续训/constant与降低LR | CMUMOSI | 66; 67; 68 | unknown | 3 | 120 | {"results_present_completion_unknown": 3} |
| osram_nested_four_dataset_per_rate_20261012 | Nested只读消融/梯度 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_gap_residual_gate_20261010 | Nested解码残差Gate/固定系数 | CMUMOSI | 66; 67; 68 | unknown | 3 | 72 | {"results_present_completion_unknown": 3} |
| osram_nested_gap_t_residual_gate_20261010 | Nested解码残差Gate/固定系数 | CMUMOSI | 66; 67; 68 | unknown | 3 | 72 | {"results_present_completion_unknown": 3} |
| osram_nested_groups1_dim512_20261008 | 原始Nested GNN及结构消融 | CMUMOSI | 66 | unknown | 1 | 26 | {"results_present_completion_unknown": 1} |
| osram_nested_input_gradient_20261010 | Nested只读消融/梯度 |  |  |  | 0 | 192 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_local8_20261006 | 原始Nested GNN及结构消融 | CMUMOSI | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_nested_local_lr_warmup_random_20261010 | Nested训练/LR/epoch/跨数据集 | CMUMOSI | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_nested_low_lr_extend150_20261010 | Nested训练/LR/epoch/跨数据集; 100→150轮完整续训/constant与降低LR | CMUMOSI | 66; 67; 68 | unknown | 4 | 128 | {"results_present_completion_unknown": 4} |
| osram_nested_mlp256_20261008 | Nested容量/Flat参数重分配 | CMUMOSI | 66 | unknown | 1 | 46 | {"results_present_completion_unknown": 1} |
| osram_nested_residual_scale_20261012 | Nested解码残差Gate/固定系数 |  |  |  | 0 | 1200 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_root_probe_20261011 | Nested Rooted concat独立读出 |  |  |  | 0 | 1076 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_rootaware_20261005 | 原始Nested GNN及结构消融 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 33 | 263 | {"complete_declared": 10, "configuration_or_metadata_only": 20, "results_present_completion_unknown": 3} |
| osram_nested_slot_delta_off_20261011 | Nested只读消融/梯度 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nested_sweep_20261005 | 原始Nested GNN及结构消融 | CMUMOSI | 66 | unknown | 10 | 160 | {"results_present_completion_unknown": 10} |
| osram_nested_training_gradients_20261010 | Nested只读消融/梯度 | CMUMOSI | 66 | unknown | 2 | 48 | {"results_present_completion_unknown": 2} |
| osram_new40_20261004 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_no_aux_cfg84_20260919 | Regression-only JEPA/MMoE/simple masked-mean目标对照 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_no_aux_cfg84_cyclic_no0_20260920 | 训练缺失分布：Uniform-forced-Text / cyclic去miss0 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 240 | {"results_present_completion_unknown": 15} |
| osram_nojepa_capacity2400_200_20260919 | Flat cfg84大容量/2400宽/200轮派生 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_nojepa_capacity2400_200_20260919_retry | 未归类/见原文 | CMUMOSI | 66; 67; 68 | unknown | 3 | 48 | {"results_present_completion_unknown": 3} |
| osram_nojepa_capacity2400_20260919 | Flat cfg84大容量/2400宽/200轮派生 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_nps_local_20261009 | Neural Production | CMUMOSI | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_nps_local_w256_20261009 | Neural Production | CMUMOSI | 66 | unknown | 1 | 16 | {"results_present_completion_unknown": 1} |
| osram_old_missing_cleanup_20261022 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_out1000_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_out700_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_out900_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_paired_aonly_localization_20261011 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 192 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_paired_history_rho025_20261002 | 双视图task loss权重rho扫参及三种子 | CMUMOSI | 66; 67; 68 | conversation-mask-v2; unknown | 17 | 256 | {"complete_declared": 1, "results_present_completion_unknown": 16} |
| osram_paired_history_views_20261001 | 双视图 task-only / CoMM-style paired history; 双视图 + symmetric InfoNCE | CMUMOSI | 66 | conversation-mask-v2 | 6 | 108 | {"results_present_completion_unknown": 6} |
| osram_paired_meld_iemocap6_20261018 | 双视图task-only迁移MELD/IEMOCAP6 | IEMOCAPSix; MELD | 66 | gcnet-random-mask-v1 | 24 | 720 | {"complete_declared": 24} |
| osram_paired_rho010_fixed_20261011 | 未归类/见原文 |  |  |  | 0 | 6 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_paired_rho010_infonce_20261011 | 双视图 + symmetric InfoNCE | CMUMOSI | 66 | conversation-mask-v2 | 1 | 22 | {"results_present_completion_unknown": 1} |
| osram_paired_rho_sweep_20261002 | 未归类/见原文 | CMUMOSI | 66 | conversation-mask-v2 | 10 | 160 | {"results_present_completion_unknown": 10} |
| osram_paired_rho_three_seed_20261002 | 未归类/见原文 | CMUMOSI | 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_paired_text_drift_20261002 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_pam_address_text_20260918 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_pam_episodic_text_20260917 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_pam_text_20260917 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_pattern_emotion_20260918 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 20 | 320 | {"results_present_completion_unknown": 20} |
| osram_persistent_patterns_20261018 | 持续/七种组合/LocalBaseGap推理评价 |  |  |  | 0 | 2027 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_post_temporal_connection_review_20261022 | 历史读出/高分审查 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_pre_v3_checkpoint_cleanup_20261022 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_predictor_reaudit_20260909 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_priority40_20261004 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_probe_direction_20261009 | Frozen probes/历史干预/梯度与12格审计 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_query_no_availability_mosi_20260907 | 早期write_step1→.6训练及三种query/write参数衍生 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_r12_3seed_20261005 | 未归类/见原文 | CMUMOSI | 67; 68 | unknown | 2 | 32 | {"results_present_completion_unknown": 2} |
| osram_r4_20261005 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout20_20261003 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout20_reconsidered_20261003 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout_interpolation_20260910 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout_path_audit_20261009 | 历史读出/高分审查 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout_paths_20260930 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout_retrospective_20261011 | 历史读出/高分审查 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_readout_top3_3seed_20261005 | 未归类/见原文 | CMUMOSI | 67; 68 | conversation-mask-v2 | 6 | 96 | {"results_present_completion_unknown": 6} |
| osram_reg_only_20260916 | Regression-only JEPA/MMoE/simple masked-mean目标对照 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_reg_only_20260919 | Regression-only JEPA/MMoE/simple masked-mean目标对照 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_reg_only_cfg84_20260919 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_reg_only_hparam_sweep_20260919 | Regression-only JEPA/MMoE/simple masked-mean目标对照 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_reg_only_hparam_sweep_20260919_100 | 未归类/见原文 | CMUMOSI | 66 | unknown | 70 | 1120 | {"results_present_completion_unknown": 70} |
| osram_relation_dual_readout_20261003 | Current–History Relation | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 4 | 64 | {"results_present_completion_unknown": 4} |
| osram_relation_dual_readout_check_20261003 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_relation_dual_readout_multiseed_20261003 | 未归类/见原文 | CMUMOSI | 67; 68 | conversation-mask-v2 | 2 | 32 | {"results_present_completion_unknown": 2} |
| osram_result_environment_inventory_20261016 | Missing协议纠正、Test-best UA恢复与环境对账 |  |  |  | 0 | 262 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_retention_20260909 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_scalar_12_source_audit_20261020 | 历史来源/组合/Query决策冻结诊断 | CMUMOSEI; CMUMOSI |  | unknown | 2 | 102 | {"diagnostic_complete": 2} |
| osram_scalar_router_candidate_ceiling_20261020 | 历史来源/组合/Query决策冻结诊断 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_second_block_transfer_survey_20261017 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_shift_capacity_20261002 | Memory-Shift-Residual / scalar filter | CMUMOSI | 66; 67; 68 | conversation-mask-v2 | 10 | 160 | {"results_present_completion_unknown": 10} |
| osram_shift_d3w256_three_seed_20261003 | 未归类/见原文 | CMUMOSI | 67; 68 | conversation-mask-v2 | 2 | 32 | {"results_present_completion_unknown": 2} |
| osram_small_flat_20261009 | Flat容量 / 小Flat+模块; Flat cfg84大容量/2400宽/200轮派生 | CMUMOSI | 66 | unknown | 3 | 72 | {"results_present_completion_unknown": 3} |
| osram_small_flat_3seed_20261009 | Flat容量 / 小Flat+模块; Flat cfg84大容量/2400宽/200轮派生 | CMUMOSI | 67; 68 | unknown | 4 | 96 | {"results_present_completion_unknown": 4} |
| osram_small_flat_nested_mlp256_20261009 | Nested容量/Flat参数重分配 | CMUMOSI | 66 | unknown | 1 | 24 | {"results_present_completion_unknown": 1} |
| osram_small_flat_nested_mlp512_20261009 | Nested容量/Flat参数重分配 | CMUMOSI | 66 | unknown | 1 | 24 | {"results_present_completion_unknown": 1} |
| osram_source_relation_attention_20261021 | SourceRelation Base/Gap均值包 | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1 | 5 | 420 | {"complete_declared": 2, "failed": 1, "smoke_complete": 2} |
| osram_source_role_direct_cross_20261020 | Base-Gap Partner Source Role | CMUMOSEI; IEMOCAPSix; MELD | 66; 67; 68 | gcnet-random-mask-v1 | 24 | 2735 | {"complete_declared": 21, "smoke_complete": 3} |
| osram_source_role_tokens_20261021 | 12-source-role tokens重做 | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1 | 4 | 290 | {"cancelled_by_user": 1, "complete_declared": 1, "smoke_complete": 2} |
| osram_source_role_v3_20261021 | V3 Hierarchical Source–Role | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1 | 4 | 380 | {"complete_declared": 2, "smoke_complete": 2} |
| osram_supervised_teacher_20260914 | 监督预训练Frozen Teacher替代EMA目标 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 20 | 267 | {"configuration_or_metadata_only": 5, "results_present_completion_unknown": 15} |
| osram_tasr_20261018 | TASR Gap来源轨迹校准 | CMUMOSI; IEMOCAPSix; MELD | 66; 67; 68; 69; 70 | unknown | 42 | 1331 | {"complete_declared": 22, "configuration_or_metadata_only": 20} |
| osram_tasr_joint_20261019 | TASR Gap来源轨迹校准 | MELD | 66 | gcnet-random-mask-v1 | 3 | 149 | {"complete_declared": 2, "smoke_complete": 1} |
| osram_tasr_lr_20261019 | TASR Gap来源轨迹校准 | IEMOCAPSix; MELD |  | unknown | 18 | 2094 | {"complete_declared": 18} |
| osram_tasr_random_only_20261019 | TASR Gap来源轨迹校准 | MELD |  | unknown | 1 | 209 | {"complete_declared": 1} |
| osram_text_anchor_20260906 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 10 | 80 | {"results_present_completion_unknown": 10} |
| osram_text_anchor_20260906_retry | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 40 | {"results_present_completion_unknown": 5} |
| osram_text_condition_optimization_audit_20261022 | Frozen probes/历史干预/梯度与12格审计 | CMUMOSEI | 66 | gcnet-random-mask-v1 | 1 | 340 | {"failed": 1} |
| osram_text_core_20260918 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 155 | {"results_present_completion_unknown": 5} |
| osram_text_path_router_20261019 | BaseText/GapText候选Router | IEMOCAPSix; MELD |  | unknown | 8 | 628 | {"complete_declared": 4, "smoke_complete": 2, "smoke_failed": 2} |
| osram_text_path_router_3seed_20261019 | BaseText/GapText候选Router | IEMOCAPSix; MELD | 67; 68 | gcnet-random-mask-v1 | 8 | 2654 | {"complete_declared": 8} |
| osram_text_path_router_joint_20261019 | BaseText/GapText候选Router | IEMOCAPSix; MELD | 66 | gcnet-random-mask-v1 | 12 | 2164 | {"complete_declared": 8, "smoke_complete": 4} |
| osram_text_path_router_joint_mosi_20261019 | BaseText/GapText候选Router | CMUMOSI | 66 | unknown | 1 | 0 | {"smoke_failed": 1} |
| osram_text_subspace_20260915 | Missing-Text predictable subspace |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_text_subspace_20260916 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_uniform_forced_text_20260918 | 训练缺失分布：Uniform-forced-Text / cyclic去miss0 | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 5 | 80 | {"results_present_completion_unknown": 5} |
| osram_utility_text_retrieval_20260918 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_v3_120_20261004 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_v3_classification_20261021 | V3 Hierarchical Source–Role | IEMOCAPFour; IEMOCAPSix; MELD | 66 | gcnet-random-mask-v1 | 14 | 2404 | {"complete_declared": 11, "smoke_complete": 3} |
| osram_v3_connection_options_20261022 | V3三种连接位置 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_v3_connection_screen_20261022 | V3三种连接位置 | CMUMOSEI; CMUMOSI | 66 | gcnet-random-mask-v1; unknown | 13 | 940 | {"complete_declared": 6, "smoke_complete": 6, "smoke_failed": 1} |
| osram_v3_paired_interfaces_20261022 | V3六组成对接口 | CMUMOSI | 66 | gcnet-random-mask-v1 | 12 | 920 | {"complete_declared": 6, "smoke_complete": 6} |
| osram_v3_source_attention_audit_20261022 | 历史来源/组合/Query决策冻结诊断 | CMUMOSEI; CMUMOSI; IEMOCAPFour; IEMOCAPSix | 66 | unknown | 4 | 0 | {"complete_declared": 4} |
| osram_v3_three_seed_20261021 | V3 Hierarchical Source–Role | CMUMOSEI; IEMOCAPFour; IEMOCAPSix; MELD | 67; 68 | gcnet-random-mask-v1 | 24 | 3176 | {"complete_declared": 24} |
| osram_v64_20260906 | 早期OSRAM容量/Head/KeyValue/output维度 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 15 | 120 | {"results_present_completion_unknown": 15} |
| osram_write_intervention_20260909 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 |  |  |  | 0 | 345 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_write_state_20260914 | WSC / Write-State JEPA | CMUMOSI | 66; 67; 68; 69; 70 | conversation-mask-v2 | 15 | 240 | {"results_present_completion_unknown": 15} |
| osram_write_step_grid_20260909 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 |  |  |  | 0 | 450 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_write_step_range_20260909 | 历史write-step冻结覆盖/保留与alpha-beta-head敏感性 |  |  |  | 0 | 600 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| osram_write_step_train_20260909 | 早期write_step1→.6训练及三种query/write参数衍生 | CMUMOSI; IEMOCAPFour | 66; 67; 68; 69; 70 | conversation-mask-v2 | 10 | 180 | {"results_present_completion_unknown": 10} |
| osram_xca_direct_20261010 | 多机制去输入残差 | CMUMOSI | 66 | unknown | 4 | 64 | {"results_present_completion_unknown": 4} |
| probe-direction-check-ZqynZo | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| protocol_debug_20260819 | 未归类/见原文 |  |  |  | 0 | 4 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| r4_scores_20261020T124720Z | 未归类/见原文 | CMUMOSI | 66 | unknown | 4 | 64 | {"complete_declared": 4} |
| relation_order_audit_20260905 | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| remote_versions_cleanup_20261022T010619Z | 未归类/见原文 |  |  |  | 0 | 656 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| roberta_prefix_jepa_20260806 | 未归类/见原文 |  |  |  | 0 | 28 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| roberta_prefix_jepa_20260806_jepa_joint_v1 | 未归类/见原文 |  |  |  | 0 | 56 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| sam_mosi_backbone_20260901 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 5 | 10 | {"configuration_or_metadata_only": 5} |
| small-flat-check-ZiVXD6 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| small-nested-wide-check-a9QNaW | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| small-nested512-check-FNvBmM | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| smoke_checkpoint_cleanup_20261022T011328Z | 未归类/见原文 | CMUMOSI | 66 | gcnet-random-mask-v1 | 1 | 1788 | {"smoke_complete": 1} |
| storage_cleanup_20261022T004254Z | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
| task_aligned_evidence_red | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 30 | 215 | {"complete_declared": 10, "configuration_or_metadata_only": 20} |
| text_anchor_mosi_20260901 | 未归类/见原文 | CMUMOSI | 66; 67; 68; 69; 70 | unknown | 10 | 20 | {"configuration_or_metadata_only": 10} |
| v3_module_test_4mEEgt | 未归类/见原文 |  |  |  | 0 | 0 | 仅文档/汇总/无叶级运行记录，不能推断未训练 |
