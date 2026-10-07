# 原模型历史诊断索引

检索日期：2026-10-07；检索时 HEAD：`d0c4bf9`。

本次只检索和整理当前工作树的已有报告，没有训练、重新推理或更改模型。数值引用报告，不声称本次重新核验全部原始预测。沿用各报告的 checkpoint 选择协议；多数为 INTERNAL DIAGNOSTIC ONLY / Test-oracle。

“条目”是分析问题，不是独立实验次数。同一批预测的后处理、同一个三阶段诊断、同一路线的多个报告不能当作独立重复证据。当前工作树未归档的远程结果及其他历史分支不在本次完整性保证内。

## A. 直接针对原 cfg84 no-JEPA 的 14 个分析条目

以下均不是新模块训练实验。A01–A04、A05–A06、A08–A09、A12–A14 是冻结推理干预或观察；A07、A10–A11 是已有记录的离线分析。均有完成报告。

| ID | 问题与实际操作 | 范围及已报告发现 | 不能推出什么 | 证据 |
|---|---|---|---|---|
| A01 | 七种固定可见组合 A/T/V/AT/AV/TV/ATV | MOSI 5 seeds；整段固定 mask；Text-present macro 86.493，no-Text macro 52.317 | 不是分别训练七个模型；不是 random-missing 子群；不能证明 A/V 输入信息上限 | [固定组合](../../experiments/osram_cfg84_fixed_modality_ablation_20260923/RESULT.md) |
| A02 | 六种整段缺失下关闭对应 active Gap 分类输入 | 5 seeds；同一次 Memory scan，保留 Base/Local；各条件平均 on−off 绝对值均 <0.1 pp；缺 T 为 −0.034 pp | 不等于 Gap 全零、预测不变或整个 Memory 无效；不是删掉 query | [persistent Gap](../../experiments/osram_cfg84_persistent_gap_ablation_20260925/RESULT.md) |
| A03 | 六种整段缺失下关闭 Base 分类输入 | 3 seeds；无 history-query adapter、无 50/50；Base 保留收益在 AV/A/V 条件分别 +4.528/+3.846/+3.540 pp，seed 波动较大 | 不是关闭全部 Memory；不是重训 no-Base；不能把均值当显著性 | [persistent Base](../../experiments/osram_cfg84_original_base_20260928/RESULT.md) |
| A04 | 同时关闭 Adapter 内 Local 和完整 Local Skip 输出 | 3 seeds ×8 rates；80.559→59.471；high 75.594→58.949 | 不是无 Local 重训成绩；当前信息仍影响 query/写入，因此不是纯历史模型 | [no Local](../../experiments/osram_cfg84_no_local_20260929/RESULT.md) |
| A05 | Base 和 masked Gap 同时缩放 α=0/.5/1 | 3 seeds ×8 rates；均值 76.593/79.961/80.559；完全关闭历史 high 降 6.278 pp | α 是读值保留系数，不是内部遗忘参数；不证明动态 Gate 必须有益 | [历史粗扫](../../experiments/osram_cfg84_history_scale_20260929/RESULT.md) |
| A06 | 同样的历史输入细扫 α=.8/.85/.9/.95/1/1.05/1.1/1.15/1.2 | 216 组；扫描最高 .95=80.612，原 80.559；差值仅 +.053 pp | 不是训练微调；不把扫描最好值当稳定机制收益 | [历史细扫](../../experiments/osram_cfg84_history_scale_fine_20260929/RESULT.md) |
| A07 | 用 A06 已有预测统计纠错/致错，以及标签辅助逐样本 oracle | 原错误 3,044 次出现，其中 150 可纠正（4.928%）；oracle 80.559→81.520；同组合中观察到不同增减需求 | oracle 不可部署；150 是重复 seed/rate 出现次数，不是 150 个独立样本；不是 Gate 可获得的收益 | [纠错空间](../../experiments/osram_cfg84_history_scale_fine_20260929/prediction_analysis/RESULT.md) |
| A08 | Adapter Local α、Local Skip β、Memory μ 的 13 组路径缩放 | 3 seeds ×8 rates ×2 splits ×13；报告使用 train/validation 记录；没有一组非原设置在所有 3 seeds 上都提升所报告 validation W-F1；统一 Adapter 输入缩放被 LN 近似抵消 | 不能把所有 13 设置当独立机制；该报告不是全测试集结果 | [读出路径](../../experiments/osram_readout_paths_20260930/RESULT.md) |
| A09 | 当前观测相同，历史额外删 A/T/V/mixed；逐层测 drift | seed66 ×8 rates；Local max 差 1.907e−6；test 历史删 T：prediction shift .06894、flip 3.58689%、52 致错/40 纠错；另有 matched anchors 和距离分层 | 不是全测试集指标；cosine/flip 变化不自动代表有害；最近删除距离不是单次扰动距离 | [历史 drift](../../experiments/osram_history_drift_20261002/summary/RESULT.md) |
| A10 | Context Audit：同 checkpoint 的历史关闭/保留逐句比较 | 3 seeds ×8 rates，离线复用；1,331 纠错/689 致错；near Local 区域 714/395；按当前位置、availability、标签强度、Local margin 分组 | MSE help 不等于类别纠错；不是分别训练 Local-only 和 Memory-on；标签分组不能部署 | [Context Audit](../../experiments/osram_context_audit_20261003/results/RESULT.md) |
| A11 | A10 补充相邻同/异极性 ×固定 Local 边界四格 | near/same：Memory 增益 +24.675 pp；near/opposite：−16.330 pp；同 availability 分层仍观察到方向差异 | 相邻极性不是实际 emotion shift；上一句标签不是 Memory 状态；不是因果证明 | [相邻四格](../../experiments/osram_context_audit_20261003/adjacent_results/RESULT.md) |
| A12 | Gap increment Stage1：同一 Full checkpoint 三读出 P_L/P_B/P_F，以及 rescue/harm observables | seed66 ×8 rates；74.820/79.439/81.068；Gap +1.629 pp，high +2.487 pp；250 rescue/155 harm；简单 norm/cosine AUROC 弱或条件依赖 | 与分别重训 local-base 消融不同；不能据此拟合可靠性 Gate；重复 rates 不独立 | [三阶段总报告](../../experiments/osram_gap_increment_audit_20261003/RESULT.md)、[Stage1](../../experiments/osram_gap_increment_audit_20261003/analysis/RESULT.md) |
| A13 | Gap increment Stage2：分类输入端逐个屏蔽 64d forward head | 同 Full checkpoint；分别屏蔽 Base head 或所有 active Gap 同 head；head7 Gap overall +.160 pp、high +.640 pp，但逐 rate 符号会变化 | 不是删 Memory head、改 query 或重训；贡献不必相加；未证明跨 checkpoint 稳定专门化 | [Head 分析](../../experiments/osram_gap_increment_audit_20261003/head_analysis/RESULT.md) |
| A14 | Gap increment Stage3：每 active Gap/head 的 query 与 read 余弦 | raw Base/Gap query .967；raw Gap/residual query .760；Base/Gap read .633；A/T/V、head、rate 明细齐全 | raw query 在 residual addressing 前；不能把 query→read 余弦差归因于 Memory 映射；余弦高不等于同信息 | [Query/Addressing](../../experiments/osram_gap_increment_audit_20261003/query_analysis/RESULT.md) |

A12–A14 是同一诊断的三个阶段：每阶段 8 次 scans，共 24 次；各阶段原三读出 CSV 相同。A10–A11 是同一 Context Audit 的两轴。A06–A07 是同一 216 组预测的推理和离线后处理。

## B. 更早的原模型/含 JEPA 模型诊断

这些是真实已有分析，但不能统称为当前 `cfg84 no-JEPA output1600/KV64`。原 checkpoint 的宽度、目标、因果性和选点规则须以各报告为准。尤其 20260910 cross-substitution 使用历史 **eight-rate-mean Test-selected best.pt**，不是后来的 per-rate BEST。

| 路线 | 操作与已报告结论 | 模型/范围边界 | 证据 |
|---|---|---|---|
| Base/Gap copy substitution | 保留 donor，将 Base 内容复制到 Gap 或反向；恰好一个模态缺失子群；相对 no-Gap，B→G 平均恢复 +.683 pp | 早期 causal η=.6 Flat，5 seeds；输出替换不是语义等价证明 | [copy](../../experiments/osram_cross_substitution_20260910/results/RESULT.md) |
| Base/Gap move 与 swap | 移动时 donor 清零；互换平均相对 normal +.024 pp，连续值仍会变化 | 与 copy 同一模型族，不是新训练模型 | [move/swap](../../experiments/osram_cross_substitution_20260910/move_swap_results/RESULT.md) |
| 内容×槽位 2×2 分解 | 已保存输出离线分解 content、slot、interaction；交互 energy share 3.64%，忽略交互 sign disagreement .29% | signed regression margin；不是 F1 增益、不是独立模块定位 | [factorial](../../experiments/osram_cross_substitution_20260910/factorial_results/RESULT.md) |
| Base/Gap 固定和插值 | U=(B+G)/2、D=(B−G)/2，改变差分，保持 B+G；测 odd/even 输出敏感性和翻转 | 5 seeds、one-missing 子群；近似不敏感不等于内部没有关系 | [interpolation](../../experiments/osram_cross_substitution_20260910/interpolation_results/RESULT.md) |
| Base/Gap 几何幅度 | norm、差/和范数、raw/centered cosine、relative difference；AT/AV/TV 的 raw cosine .646/.650/.314 | **该早期 context=512，包括零 backward half**，不是当前 1024 的 forward512；不是 CKA | [magnitude](../../experiments/osram_cross_substitution_20260910/magnitude_results/RESULT.md) |
| 写入与历史关联保留 | replay 当前 decay 前后及 write 前后，读历史真实 key/value probe；区分旧关联保留和当前拟合 | 有早期 retention，也有 cfg84 MOSI 20260920 replay；probe fit 不等于情感价值 | [早期 retention](../../experiments/osram_retention_20260909/RESULT.md)、[cfg84 replay](../../experiments/osram_memory_replay_20260920/RESULT.md) |
| 冻结写入保护 | protected write vs reference/global norm-matched；pilot 5-rate +.1185 pp，但比 global −.0102 pp；另有 5 seeds 四模式完整表 | IEMOCAP4 历史 forward-only 模型；不等于 cfg84 MOSI / Nested；降低保留误差不保证 W-F1 提升 | [pilot](../../experiments/osram_write_intervention_20260909/RESULT.md)、[full5](../../experiments/osram_write_intervention_20260909/full5/RESULT_FULL5.md) |
| Gap-query identity swap | MOSI cfg84 Full seed67；只交换两个同时 active 的缺失 query，Memory trajectory 相同；.5/ .7 overall −.292/+ .162 pp | 同一个 best_miss_.7 checkpoint 跑四 rates；不是 A14 的 cosine；数值移动通常不翻分类符号 | [query swap](../../experiments/osram_gap_query_swap_20260920/RESULT.md) |
| 初始 latent 样本对应诊断 | regression/contrastive 分支分开；centered cosine、shuffle gap、retrieval、rank、variance；reg rank 7–9、retrieval 接近 chance | 20260831 Slot Missing-M3，不是 OSRAM no-JEPA；原报告归因/措辞不能自动视为已证明的普遍机制 | [latent](../../experiments/missing_m3_mosi_latent_diagnostic_20260831/EXPERIMENT.md) |
| η=.6 predictor 复查 | 5 seeds natural miss=.5 A/V/AV；Text raw cosine 约 .93，centered 约 .03–.04；相比 η=1 没有修复样本对应 | 早期带 predictor 的 OSRAM；仅冻结评估，prediction 不在当时分类路径回灌 | [reaudit](../../experiments/osram_predictor_reaudit_20260909/RESULT.md) |
| Complete-State JEPA target/prediction/gradient audit | full target ridge 可读；预测弱且低 rank；共有参数梯度有少数冲突，但加权 state 梯度很小 | 5 State-JEPA checkpoint、miss=.5、output700；拟合 ridge，不是训练 backbone；不是 cfg84 | [State audit](../../experiments/osram_complete_state_20260910/AUDIT.md) |
| Teacher 信息出口 | Audio/Text/Visual projector、fused node、hidden 的固定 ridge；Text85.55，Audio53.07，Visual55.70 | 5 supervised Teachers；不是原始 A/V 特征信息上限 | [information](../../experiments/osram_supervised_teacher_20260914/INFORMATION_AUDIT.md) |
| Teacher Text probe transfer | 同一 target-space probe 读真实 vs predicted Text，加 prototype/shuffle 控制；真实 Text 84–87，predicted 约47–55 | 小样本 natural A/V/AV，5 frozen Teacher/Student；不是 cfg84 Fixed-pattern Test | [transfer](../../experiments/osram_supervised_teacher_20260914/TEXT_TRANSFER_AUDIT.md) |
| Text 坐标对齐/预测可读性 | 在 predicted Text 上新拟合 ridge，或 label-free affine 对齐后用原 probe；没有恢复到真实 Text 水平 | affine 比正交旋转更一般；有限线性 probe 不证明“没有信息” | [alignment](../../experiments/osram_supervised_teacher_20260914/TEXT_ALIGNMENT_AUDIT.md) |
| Text 任务方向可达性 | 把固定 Text ridge 转为 raw latent 方向，比较 MMoE 与 observed-input ridge 对方向的预测 | 不是 256d 的数学信息上限；train 拟合高、所报告 validation 对应弱 | [direction](../../experiments/osram_supervised_teacher_20260914/TASK_DIRECTION_AUDIT.md) |
| MMoE target gradient conflict | 目标共现；共享 experts 分目标梯度 cosine/norm；42 对中14负，Visual norm较大 | 旧 `train_rate_mode=all`，seed66 epoch44；不是当前 cyclic/no-JEPA，不证明分类受损原因 | [conflict](../../experiments/missing_m3_target_ple_20260903/TARGET_CONFLICT_DIAGNOSTIC.md) |
| MOSI bottleneck 综合定位 | predictor 是否回灌、classification/JEPA 梯度、既有 completion 证据联合核对 | 历史诊断总结，不额外重复计为一次独立实验 | [bottleneck](../../experiments/mosi_bottleneck_diagnosis_20260907/DIAGNOSIS.md) |
| 异步模态历史 ridge | local/generic/async/shuffle/random-history；高缺失部分相关性收益，完整输入下降 | 早期 Text-anchor Student，包含前后观测；不是 causal OSRAM；仅 probes，无 SSM 正式训练 | [async](../../experiments/asynchronous_state_audit_20260905/RESULT.md) |
| Temporal/Speaker 顺序 audit | operator commutator 与 shuffle；一阶 vs ordered ridge；MOSI 不支持顺序增益 | 早期 GCNet windowp/f=2，含双向邻域；不是 Nested 的 evidence-head 图 | [relation order](../../experiments/relation_order_audit_20260905/RESULT.md) |

## C. 后续版本的失败诊断：不能叫原 cfg84 诊断

| 分析 | 实际对象与发现 | 证据 |
|---|---|---|
| Post-GRN residual bypass | 3 seeds ×8 rates；GRN on79.521/bypass78.978，on 平均有益但整套模型仍低；还检查 dropout/RNG、初始化跨设备一致性 | [GRN diagnosis](../../experiments/osram_cfg84_post_grn_20260929/DIAGNOSIS.md) |
| 双视图 A/B 同 mask 历史删 Text | Flat/A/B 的同 anchor 比较；B 翻转更少（1.802% vs3.587%），但原历史 anchor W-F1 不更高；当前用户不要求重跑 Flat 对照 | [paired drift](../../experiments/osram_history_drift_20261002/paired_summary/RESULT.md) |
| 双视图 loss 大小 | 200 epoch logs 离线复算；task 对全部有效 utterance，InfoNCE 只对合格 anchors；B 后期加权 InfoNCE/task 比率较高 | [loss audit](../../experiments/osram_paired_history_views_20261001/loss_summary/RESULT.md) |
| Relation near-only splice | 保存预测，固定 abs(原 Local-only)<=.25 切换；seed66 81.068→81.270；不是新训练或 gold 同/异极性切换 | [splice](../../experiments/osram_current_history_relation_20261003/near_splice_results/RESULT.md) |
| Relation residual-off 分解 | F/R-off/R-on=81.068/80.361/80.576；branch +.216，但 trained original path −.707；不是 Nested | [residual-off](../../experiments/osram_current_history_relation_20261003/residual_off_analysis/RESULT.md) |

## D. 容易误记成“诊断”的重训实验

- [local-only/local-base/raw-gap/full](../../experiments/osram_mosi_memory_gap_ablation_20260920/summary/RESULT.md)：四版本 **分别从头训练**，不是 A12 三读出；5seed 77.797/79.728/80.118/80.445。
- [三数据集 dual-projector completion](../../experiments/osram_joint_dual_projector_completion_20260922/RESULT.md)：预训练后下游 **重新训练**；不是仅冻结推理补全干预；配对3seed overall −1.718 pp。
- [pretrained 固定组合评测](../../experiments/osram_joint_pretrained_fixed_modality_ablation_20260923/RESULT.md)：对上项训练完成模型的 evaluation-only；no-Text macro53.752→52.070，不能只取 A-only+7.711 说整体成功。
- History-query、50/50 conversation mix、Gate、memory-only Adapter、decision correction、Complete-State/WSC 等有独立训练；训练结果本身不是冻结 readout diagnostic。
- [Future-State](../../experiments/osram_future_state_20260914/RESULT.md)：报告仅实现与一更新 smoke，没有正式训练/F1结果；不能计为完成性能诊断。
- [Nested sweep/code audit](../../experiments/osram_nested_sweep_20261005/CODE_STORY_AUDIT.md)：已确定 Nested 的结构审计和训练消融；不是原 cfg84 query/readout 历史诊断。
- GPU health、protocol debug、单步有限梯度 smoke 属工程检查，不能计作任务机制证据。

## E. 余弦与读出：究竟已经测到哪一层

1. predicted-vs-target latent：B 中 latent/reaudit/Teacher 系列；不适用于 no-JEPA 的实际分类输入。
2. Base-vs-Gap 几何：早期 cross-substitution magnitude；有 raw/centered cosine，但 context/checkpoint 与 cfg84 不同。
3. 原 cfg84 query/addressing/read：A14；raw query cosine 与 residual query cosine 必须分开。
4. 原 cfg84 read→decision：A12/A13、整段 A02/A03、缩放 A05/A06/A08、无 Local A04。
5. 历史变化→Local/read/hidden/prediction：A09；严格相同当前 anchors。
6. 逐句标签辅助分析：A07/A10/A11；仅离线现象，不能当输入或决策规则。
7. **本次定位到的原模型诊断中，没有发现已完成的最终旧版 Nested“输入 token→rooted summaries→δL/δB/δG→Adapter”逐层关系诊断报告。**这是当前索引的缺口，不是断言远程/其他分支绝不存在该产物，也不是自动启动新实验的指令。

## 使用原则

用户已固定 Nested 和 Test 选模协议，本索引不提出更换模型或选模规则。后续分析应优先复用原有问题和工具，不反复重做已完成实验；需要新分析时明确它比已有报告新增了哪一层证据。原 cfg84 结论不能无检查地搬到 Nested，early JEPA 结论更不能套到 no-JEPA。失败诊断保留，但不得据其中某个表格预先指定 Nested 的机制。

本次交付仅此检索索引；未修改、覆盖原报告/预测。外部文献无需新增；结果统计来源全部是本地报告。
