# V3 接口改进的三个候选：研究记录，尚未批准实施

2026-10-22；审查源码 `53d91eab5d39d870b7d1174c3cb505f8c94c7c4b`，分支 `feature/ecerc-features`。

用户本轮要求“先研究，给几个方案，还是 V3”。本记录不是最终实现规格或训练授权。没有修改模型、启动训练、GPU 推理或 SSH；没有修改任何旧实验。只实例化现有 V3 做 CPU 参数计数，并核算候选预算。

## 1. 保留什么，为什么不是继续简单加宽

三个候选都保留 V3 的来源追踪、同源 Base–Gap 配对、同角色跨来源交互，以及原 CHM 联合写入、读前写、原 Query、Local Skip、任务头、Loss 和 GCNet Missing。来源仍指联合写入中 Value 槽位的贡献，不是纯模态语义。

以下尺寸以**标准 V3-64 / PairMLP128 / Flat1600 / CHM8×64**为结构参照，不把已有 Flat512/V3-128 或 Head48 自动改标为基线。候选分别验证三个不同约束，不组合启用：

| 候选 | 主要改变 | 保持不变的关键对照 | 新模块参数预算 |
|---|---|---|---:|
| 当前 V3 | 每角色3来源 → Local CrossAttention汇总64 → Decoder512 → 加原读值 | 现状 | 335,006，现有类实数 |
| A：末端保留来源槽 | 每角色三个64维Token直接固定槽拼接192 → Decoder512 | Layer1/2、原Flat、输入残差、零出口 | 514,718，解析预算 |
| B：独立特征接入Flat隐藏层 | 四角色64维结果拼成256，投影到Flat首层1600维激活之前 | Layer1/2、末端Local读取、原Base/Gap、原Flat输出和Skip | 611,998，解析预算 |
| C：保留八个Memory Head | 在各原生64维Head内部运行V3层级，最后拼回8×64 | 关系规则、原Flat、输入残差、零出口；参数量近乎不变 | 335,518，解析预算 |

只有当前 V3 已实现；候选数字不是新模型实测，耗时尚未测量。

已有结果给出的约束：

- MOSI Flat512+V3-128 三种子八率 W-F1 为79.314544，对原大Flat79.281728仅 **+0.032816 pp**；高缺失为 **−0.339385 pp**。缩小Flat、放大V3尚未得到稳定增益，而且缺少Flat512-only的67/68对照。[归档](../osram_mosi_v3_capacity_3seed_20261022/RESULT.md)。
- MOSEI Flat1024的同批seed66对照，off → V3-256：八率83.077816→83.104360，高缺失79.849565→79.819696。不是显著改善证据。[归档](../osram_mosei_v3_capacity_20261022/RESULT.md)。
- 标准V3的MOSEI末端Attention接近均匀，query-swap权重变化很小；这是选择行为现象，**不是Attention导致性能不涨的证明**。[完整测试分布审计](../osram_v3_source_attention_audit_20261022/RESULT.md)。
- Nested concat已有实现和运行记录；方向复核中的66.315是当时76/100的中途结果，不能冒充最终150轮分数。V3 Direct本地正式结果仍未归档完成，两轮smoke不算训练完成。不用这两项推断concat或直接替换必败。[Nested记录](../osram_nested_concat_20261011/STATUS.md)、[中途复核](../osram_nested_concat_20261011/DIRECTION_REVIEW.md)、[V3 Direct记录](../osram_mosi_v3_direct_20261022/README.md)。

## 2. A：保留来源槽，而不是再汇总成一个64维出口

保留V3 Layer2输出 `Z[T,B,4,3,64]`。对每个读取角色r：

`delta_r = D_r(concat(Z[r,A], Z[r,T], Z[r,V]))`，`D_r:192→512`。

删除末端四个Local Query投影与output CrossAttention，保留Layer1中的Local条件。历史来源槽按A/T/V固定顺序，不可见历史来源用安全零，不向前挪动槽位。原Layer2仍可以交换来源内容，所以这些Token是“带来源索引的混合表示”，不是未经混合的纯来源。

输出仍为 `C'_r=C_r+delta_r`，四路送入原Flat；四个Decoder保持零初始化。无历史、ATV、padding和inactive Gap严格按当前V3旁路。Layer1/2 Norm不改，不附加新的末端Norm、Loss或Gate。

它检验的是：**是否在交给读出之前，把不同来源过早压进同一个出口，限制了关系表示的使用。** 对固定激活角色，Decoder输出差异的线性维数上限由64变为192，但不等于保留原始512维全部信息，也不意味着任务性能一定更好。

这是最小、最清楚的第一候选：新增约18万参数，同时去掉最后一次来源选择。风险是更多自由度和过拟合，而且失去该处显式Local Query；Local仍通过PairMLP条件化，但不与原末端Query等价。若有效，仍需容量对照，不能把收益自动归因于保留槽位。

**去重：**不是已做的V3 width128/256；也不是Nested把五角色图摘要320维拼到Flat。准确的新点是V3 Layer2之后的三个来源出口，不宣称concat本身有创新。

## 3. B：让V3关系特征进入Flat，而不是先写回Base/Gap坐标

保留当前V3直到四个角色的Local CrossAttention输出：`z=[z_B;z_GA;z_GT;z_GV]∈R^256`。删除四个 `64→512` Decoder，原始Base/Gap完全不做加法或替换。

设原Flat输入 `x=[L;B;GA;GT;GV]∈R^4352`，原首层为W1，末层为W2。新接口：

`p = W1 LN_x(x) + b1 + P MaskedLN_z(z)`

`h = LN_out(Skip(L) + W2 Dropout(GELU(p)) + b2)`

`P:256→1600`，无bias。原4352维LN不扩大、不删除backward零半边；z使用独立LN256，之后重新应用角色mask，并在无历史/ATV/padding时强制整个新增项为零，避免LN的affine bias产生假证据。四个角色的位置固定。LN256仍会耦合四个角色的统计并改变幅度信息，不能默认它优于无Norm；这里是候选接口的一部分，不能隐藏为“完全只移动一根线”。

这不是旧版 `Flat_anchor+SourceRelation1600`：新增表示在**原Flat首层非线性之前**参与任务融合，不在整个Adapter之后另加预测表示。也不是新增分类头或预测混合。

该写法数学上等价于分别归一化的 `[x;z]` 经一个Linear，**不能把隐藏层注入与concat包装成两个新机制**。其意义是保留关系特征的独立输入系数P，不要求它先成为原Base/Gap坐标中的修正量。

初始化需要准确区分：方案针对从头联合训练，P普通非零初始化；原Flat的W2/b2仍零，因此初始logits可保持原LocalSkip路径，但首步W2梯度已经不同。这是设计差异，不是原训练轨迹等价。它去掉新增零Decoder的启动屏障，仍保留W2的原屏障；不保证长期优化更好。P的初始影响也不保证微小，且GELU前相加会改变原支路经过非线性后的响应，不应称“原预测分量完全不变”。若加载已训练且W2非零的Flat，不能再宣称初始预测等价，需另定初始化，不能悄悄复用该承诺。

**去重及风险：**Nested concat属于近邻，其输入是不同图摘要，且改变了联合输入LN；本方案保留原LN、另给V3特征入口。已失败的concat近邻降低了乐观预期；本方案仍可能被Flat忽略或过拟合，而且保留了每角色末端64维瓶颈。不能同时再采用A，否则无法区分改变出口还是改变接入位置的效果。

## 4. C：按原Memory Head运行V3，而不是把整路512统一压成64

组织来源读值为 `[T,B,role4,source3,head8,value64]`。这里八个Head是CHM的真实Head，**不是V3 MHA内部四个attention head**。

每个Memory Head h使用自己的无bias `P_h:64→64`，在该Head内跨来源、Base/Gap共享坐标。按原V3方式分别编码同源Base–Gap，再做同角色跨来源Attention，最后Local Query汇总三个来源，得到 `[T,B,4,8,64]`。

- PairMLP、两个MHA、LayerNorm在八个Head之间共享，不复制八套完整V3。
- 新增 `head_embedding[8,64]`，初始为零，加在现有Local64条件和role Query上；PairMLP仍343维。避免默认单位方差embedding额外改变初始查询尺度。
- 每个role/head用独立 `D[r,h]:64→64`，零初始化；八个结果按原Head位置拼回512，再加原读值。
- 保留当前Norm与mask规则，禁用情况下仍原路径；不加跨Head图或额外Loss。

参数近乎相同：原 `512×64` 投影与 `8×64×64` 参数相等；原四个 `64→512` Decoder与32个 `64→64` Decoder连bias也相等，仅新增512个Head embedding参数。最终角色校正由八个64维出口组成，不再共享一个64维总出口。

它检验的是**相同关系机制是否更适合在Memory已有的分块坐标内处理**，而非“Head多就更有创新”。代价是关系/Attention处理的token批量大约8倍（并非实测总耗时8倍），也放弃了输入投影中的自由跨Head混合。没有证据证明八个Head已有稳定语义分工。

**去重：**Nested、Nested-Local8已经保留过原生Memory Head。本方案的区别仅是同时保留source轴，并在每Head内执行V3的两级关系；不能声称首次Head建模。

## 5. 文献启发与不能推导的结论

[MulT作者代码](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L101)把不同方向的Attention结果拼接后才进入后续时间编码，末端又拼接模态分支；并非所有出口都先相加。这支持把“独立出口”当作可检验的工程选择，不支持保证A/B能提升OSRAM。

[DialogueGCN原文](https://aclanthology.org/D19-1015.pdf)与[已核对代码及其他论文接线](../osram_post_temporal_connection_review_20261022/RESULT.zh.md)也提供原表示与新表示分别交给分类器的例子。它们处理的时序节点、可见未来和任务条件不同；不能将论文涨分归因为concat，更不能搬用其分数预测本实验。C主要依据当前CHM真实分块及Nested源码，不假借其他论文证明它有效。

## 6. 建议与下一步边界

按结构价值和参数可控性，优先级建议 **C → A → B**：C在近等参数下检验是否需要保留原生Head结构；A是修改最局部、计算压力较小的来源出口对照；B与已有concat近邻更接近，优先级较低。若首要目标是最低代价排查出口，则先A。C关系计算更多，不把近等参数误称近等计算。三个都是同一V3的候选变体，不是三个已验证的新方法。

如果用户决定继续，首轮只选其中一个，与标准V3和对应Flat对照；固定训练协议、旧结果及真实Mask来源。MOSI可作为快速筛选，但不能据此保证MOSEI有效。若使用既有Test-best规则，继续标记内部探索，不称单checkpoint部署成绩。当前不确定候选、epoch、种子或队列，不自动创建训练任务。

机制判断不只看范数或Attention熵：需看同配置的任务指标，若有收益再补同checkpoint开关和必要容量对照。无需在用户选方案前再启动一轮诊断。

目前明确拒绝：单纯再加宽、同时去掉所有Norm、把输入add改名concat、叠加A/B/C、恢复旧Direct/旧训练或追加新Loss。它们要么已有近邻，要么混合过多变量。

记录验证包括源码/引用路径、历史数值、参数预算和版本状态；**不包含候选实现、前后向测试、性能提升证据或实时远端实验状态**。
