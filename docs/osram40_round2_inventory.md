# OSRAM 第二轮 20 个输入侧模块：源码机制清单

本文依据 2026-10-04 本地 [ROUND2.json](../experiments/osram_meaningful20_round2_20261004/ROUND2.json)、关联候选卡及实际 `meaningful_input*.py` 整理。它是静态实现说明，不是训练完成清单、效果排名或论文复现报告。清单中的动机均为待验证假设，不能据此认定原 Flat 存在某种已证实的瓶颈，也不能继承原论文的精度、效率或理论保证。实验清单本身标为内部诊断、按缺失率自适应测试集 oracle 筛选，不构成论文效果证据。

## 共同边界与写回位置

实际入口是 [MeaningfulInputAdapter，L33](../gcnet_missing_m3/meaningful_input.py#L33)。它只接收已有扫描得到的 Local，以及 Base/Gap 的真实前向读出；不新增 OSRAM 查询、写入或跨话语缓存。默认 Local 为 256 维，每个 memory role 为 `8 × 64 = 512` 维；Gap-A/T/V 只在对应当前模态缺失时启用。合法非空 availability 下，常见 head-token 集合为 Local 1 个、Base 8 个、有效 Gap 0/8/16 个，即 9/17/25 个 token。**角色级、逐 head 坐标级和概率变量级模块不应强称为同一种 token 化。**

外层先用 `where` 清除无效位置，跳过 padding、首个有效话语及全部读出已消融为零的行；只向核心传入有效历史行。修改限于 Flat 输入的 Local 和/或已有前向 memory 坐标，后向 context 半段不由这些核心改写。实际 [OSRAM 集成，L1714–1732](../gcnet_missing_m3/osram.py#L1714) 使用 `readout_local` 拼接 adapter 输入，但 skip 始终为 **`local_skip(local)`，即原始 Local**；原 `emotion_adapter`、归一化与分类路径继续使用。新增核心没有补全目标、辅助重构/密度损失或新的任务头。

写回范围按实际 `forward` 区分：

| 范围 | 本轮方法 |
| --- | --- |
| 修改 Flat 的 Local 输入及有效 memory 输入，共 13 项 | 5 个 attention 模块；TokenLearner/Fuser、MBT、NetVLAD、ToMe；MCAN、DCN；SPDNet；PPGN |
| 仅修改 memory，Local 输入与 Local skip 均保留，共 6 项 | Spline、LRPCA、DEQ、NLM、FSPool/FSUnpool、RAT-SPN |
| 仅修改 Flat 的 Local 输入，memory 保留，共 1 项 | MAC；其 Local skip 仍保留原始 Local |

大多数模块以零初始化仿射解码器向原始槽位加修正；Spline 用恒等样条初始化，LRPCA 用零初始化分量重组系数，DEQ 用零初始化反馈矩阵形成原生恒等映射。这里记录初始化设计，不将静态检查当作数值验证或训练效果证明。

候选依据：[attention 卡](../experiments/osram_meaningful20_round2_20261004/attention_candidates.json)、[structured 卡](../experiments/osram_meaningful20_round2_20261004/structured_candidates.json)、[representation 卡](../experiments/osram_meaningful20_round2_20261004/representation_candidates.json)、[VQA 卡](../experiments/osram_meaningful20_round2_20261004/vqa_candidates.json)、[spectral 卡](../experiments/osram_meaningful20_round2_20261004/spectral_candidates.json)、[replacement 卡](../experiments/osram_meaningful20_round2_20261004/replacement_candidates.json)、[PPGN 卡](../experiments/osram_meaningful20_round2_20261004/ppgn_candidate.json)。卡中曾讨论但未进入 ROUND2 的 GroupViT、BAN、MFH、iSQRT 等不计入本清单；卡中的“待实现/待验证”历史措辞也不用于推断当前运行状态。

## 逐项清单

### 01. Differential Transformer V1 — `differential_attention_v1`

- **实现/写回：** [meaningful_input_attention.py，DifferentialAttention，L31](../gcnet_missing_m3/meaningful_input_attention.py#L31)；[AttentionInputAdapter，L220](../gcnet_missing_m3/meaningful_input_attention.py#L220) 写回 Local 与 memory。
- **实际机制：** 128 维有效 token 上使用 4 个差分 head；每个 head 计算两张独立 softmax 注意力图，再以指数差参数化的可学习 λ 做有符号相减。共享 value 聚合后逐 head RMSNorm，并乘固定 `1 − λ_init = 0.8`，最后投影；不是在普通 attention 后加一个标量 gate。
- **为何放在已有读出：** 测试两套匹配分布的差能否抵消 Base/Gap 共有的干扰，让原 Flat 接收不同的相对证据；“共有干扰”尚未被本实验确证。
- **主要迁移限制：** 没有语言模型主干、因果三角 mask 或 RoPE；符号相减同样可能消掉少量但有用的变化证据，指数 λ 也有数值风险。不宣称原论文的去噪收益。
- **来源：** [论文](https://arxiv.org/abs/2410.05258) · [作者代码（固定版本）](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/Diff-Transformer/multihead_diffattn.py)。

### 02. Flowformer — `flowformer_conservation`

- **实现/写回：** [meaningful_input_attention.py，FlowAttention，L61](../gcnet_missing_m3/meaningful_input_attention.py#L61)；共用 `AttentionInputAdapter`，修改 Local 与 memory。
- **实际机制：** sigmoid 正值 Q/K → source/sink 容量倒数 → 双向 conservation 修正 → source competition 与 sink allocation → 容量归一化的结合式 KV 聚合 → 输出投影。竞争归一化使用真正有效 token 数，而非填零后的固定长度。
- **为何放在已有读出：** 测试相关读出之间的联合资源分配，而非逐通道独立缩放，是否能给 Flat 更有区分度的上下文。
- **主要迁移限制：** 不是 Sinkhorn 传输或模态可靠性估计；memory token 数多于 Local，守恒机制不保证角色均衡。正核与归一化也可能抹平真实分歧；短集合不承接长序列效率结论。
- **来源：** [论文](https://proceedings.mlr.press/v162/wu22m.html) · [作者代码（固定版本）](https://github.com/thuml/Flowformer/blob/61f5d8c83907955a8bc226af03b81b59b2724c8f/Flow_Attention.py)。

### 03. Compositional Attention — `compositional_search_retrieval`

- **实现/写回：** [meaningful_input_attention.py，CompositionalAttention，L87](../gcnet_missing_m3/meaningful_input_attention.py#L87)；修改 Local 与 memory。
- **实际机制：** 4 套 search 注意力分别作用于 4 套 value retrieval，构成全部 16 组候选结果，再按候选内容对 retrieval 维做第二次 softmax 选择，拼接投影。保留 source 的对角排除；不是把第 s 个 Q/K 永久绑定到第 s 个 V。
- **为何放在已有读出：** 测试“按什么标准寻找相关历史”与“取回哪些属性”能否解耦，使同一 Base/Gap 匹配关系支持不同信息提取。
- **主要迁移限制：** 排除自身依靠原始输入残差保留自身信息；不迁移 source 任务解码器。多组候选增加成本，search/retrieval 也可能塌缩，不能声称已经得到语义分工。
- **来源：** [论文](https://arxiv.org/abs/2110.09419) · [作者代码（固定版本）](https://github.com/sarthmit/Compositional-Attention/blob/c2c9d8391144db4155922bb859183e844eefb568/Contextual_Retrieval_Task/model.py)。

### 04. Dual Attention Transformer — `dual_attention_symbolic_relations`

- **实现/写回：** [meaningful_input_attention.py，DualSymbolicAttention，L117](../gcnet_missing_m3/meaningful_input_attention.py#L117)；修改 Local 与 memory。
- **实际机制：** token 经可学习模板检索 8 项 symbol library；两路 sensory head 聚合普通内容，两路 relational head 以独立 selector 选择 token 对，并聚合 4 维显式关系向量和 sender symbol，最后拼接。关系比较投影与关系选择投影分离。
- **为何放在已有读出：** 测试同时保留原始内容与 Local/Base/Gap 间比较结果，能否为 Flat 提供比单一路径更有用的证据。
- **主要迁移限制：** symbol 只是内容条件的学习向量，不是真实对象、逻辑标签或缺失模态；不存在持久边状态或新的历史缓存。没有原论文关系任务的语义监督，symbol 分配可能塌缩。
- **来源：** [论文](https://proceedings.mlr.press/v267/altabaa25a.html) · [作者代码（固定版本）](https://github.com/Awni00/dual-attention/blob/dce218cbf5ec9aa7f90687c1323050a1fba17966/dual_attention/dual_attention.py)。

### 05. DCFormer — `dcformer_dynamic_head_composition`

- **实现/写回：** [meaningful_input_attention.py，DynamicallyComposedAttention，L192](../gcnet_missing_m3/meaningful_input_attention.py#L192)，配合 [_HeadComposition，L174](../gcnet_missing_m3/meaningful_input_attention.py#L174)；修改 Local 与 memory。
- **实际机制：** 对每个 token 对的 4 维 attention-head 向量，在 softmax 前后各执行独立 Compose：恒等项、query/key 条件的 rank-2 非对角 head 混合、query/key 对角项。生成因子有 RMS 归一化；softmax 后 Compose 不再强制重新归一化。
- **为何放在已有读出：** 测试匹配标准能否按接收方和发送方内容交换信息，而不是固定独立的注意力 head；这 4 个计算 head 不是 OSRAM 的 8 个物理 value head。
- **主要迁移限制：** 后置 Compose 可产生有符号、非随机矩阵权重，不能解释成概率；没有原模型的 RoPE、缓存或长序列设置。区别于第 03 项：此处混合 score/weight head，不是聚合后选择 retrieval 输出。
- **来源：** [论文](https://proceedings.mlr.press/v235/xiao24d.html) · [作者代码（固定版本）](https://github.com/Caiyun-AI/DCFormer/blob/df8671f23b4830cb3a254c23e5576d45141cfbd6/pytorch/dcformer/modeling_dcformer.py)。

### 06. 条件有理二次样条耦合 — `conditional_rational_quadratic_spline_coupling`

- **实现/写回：** [meaningful_input_structured.py，_SplineCoupling，L81](../gcnet_missing_m3/meaningful_input_structured.py#L81)；仅修改 memory，返回原 Local。
- **实际机制：** 对每个真实 head 拼接四角色为 256 维，执行 4 次交替奇偶坐标 coupling；以未变换半部、Local、availability、head one-hot 为条件，生成 8-bin 单调有理二次样条。区间为 `[-3,3]`，区间外恒等，缺失角色坐标保持零；均匀结点及单位导数构成恒等初始化。
- **为何放在已有读出：** 测试条件可逆坐标重整能否改变跨角色证据进入原 Flat 的几何关系，同时避免先做不可逆池化。
- **主要迁移限制：** 只使用正向变换，没有密度、log-det 目标或采样。可逆性只针对固定条件下的有效坐标，不代表整个 mask/OSRAM 系统可逆；范围外数值不变，也不具备主动删除干扰的能力。
- **来源：** [Neural Spline Flows 论文](https://arxiv.org/abs/1906.04032) · [作者代码（固定版本）](https://github.com/bayesiains/nsf/tree/1dcc0f2d0a0335974052d933c0edd04443eb5d14)。

### 07. Learned Robust PCA — `learned_robust_pca_evidence_decomposition`

- **实现/写回：** [meaningful_input_structured.py，_LearnedRobustPCA，L121](../gcnet_missing_m3/meaningful_input_structured.py#L121)；仅修改 memory；**Local 不进入分解核心**。
- **实际机制：** 将完整有效角色列组成 `512 × C` 矩阵，`C=1/2/3`；单列直接保留。确定性、断梯度的 rank-1 SVD 初始化后，做 14 次学习阈值的稀疏残差软阈值及逆 Gram 预条件双因子同步更新。最终 `Y_new = Y − tanh(γ) S`，γ 初始为零；不是逐 head attention 或直接标量 gate。
- **为何放在已有读出：** 测试“共享低秩部分”和“少数冲突坐标”是否具有不同任务价值，让任务损失学习保留还是压低稀疏分量。
- **主要迁移限制：** 仅 2–3 列时分解可辨识性弱；稀疏差异可能恰好是有效情绪变化。断梯度谱初始化、rank-1、数值下界及分量重组均为迁移选择，没有原恢复监督或恢复定理保证。零 γ 导致核心梯度在桥接系数离开零后才进入。
- **来源：** [论文](https://arxiv.org/abs/2110.05649) · [作者代码（固定版本）](https://github.com/caesarcai/LRPCA/tree/d241beef37f855a5bab298f58861611f8d945f7b)。

### 08. 收缩型 Deep Equilibrium — `contractive_deep_equilibrium_evidence`

- **实现/写回：** [meaningful_input_structured.py，_DeepEquilibrium，L229](../gcnet_missing_m3/meaningful_input_structured.py#L229)，隐式反传为 [_ImplicitEquilibrium，L200](../gcnet_missing_m3/meaningful_input_structured.py#L200)；仅修改 memory。
- **实际机制：** 每个真实 head 的四角色 256 维状态满足 `z = x + 0.9 m P̄ tanh(Q̄(mz) + Cc + b)`，条件含 Local、availability 与 head 身份；P/Q 用 `1 + Frobenius 范数` 归一化。前向 Picard 求固定点，反向另解转置 Jacobian 隐式伴随系统；各行独立停止，最多 50 步、容差 `1e-6`，不收敛显式报错。P 初始为零。
- **为何放在已有读出：** 测试共享非线性关系的平衡状态，是否比固定次数的普通聚合更适合协调同一 head 的角色证据。
- **主要迁移限制：** 迁移的是“固定点 + 隐式梯度”核心，不是原 DEQ 序列网络；Picard 和收缩映射替换原 Broyden 配置，无 warm start 或额外 Jacobian 损失。有限预算不等于无条件收敛，收缩限制也可能削弱表达能力。
- **来源：** [论文](https://arxiv.org/abs/1909.01377) · [作者代码（固定版本）](https://github.com/locuslab/deq/tree/1fb7059d6d89bb26d16da80ab9489dcc73fc5472)。

### 09. TokenLearner V1.1 + TokenFuser — `tokenlearner_fuser_v11`

- **实现/写回：** [meaningful_input_representation.py，_TokenLearnerFuser，L71](../gcnet_missing_m3/meaningful_input_representation.py#L71)；[RepresentationInputAdapter，L225](../gcnet_missing_m3/meaningful_input_representation.py#L225) 修改 Local 与 memory。
- **实际机制：** 输入条件的 analysis softmax 将有效集合压到 4 个 128 维 token；经过完整 Transformer，再做压缩 token 轴的学习混合；另一套输入条件 sigmoid synthesis 将结果分别送回每个原 token，最后零仿射桥加回原槽位。analysis/synthesis 不互绑。
- **为何放在已有读出：** 测试少量共享证据基底经过交互后，能否为每个 role/head 生成不同修正，而不改变原 Flat 的输入布局。
- **主要迁移限制：** 使用作者 V1.1 MLP 分支，不伪造图像卷积网格或视频时间轴。token mixer 从源零初始化改为 Xavier，避免与零解码桥形成双零死分支；两个映射不是逆映射，不能声称重建缺失内容或已获得加速。
- **来源：** [论文](https://arxiv.org/abs/2106.11297) · [作者代码（固定版本）](https://github.com/google-research/scenic/blob/8c113c501c9f700b69899c55a69e65bb46727da6/scenic/projects/token_learner/model.py)。

### 10. Multimodal Bottleneck Transformer — `mbt_role_bottleneck`

- **实现/写回：** [meaningful_input_representation.py，_MultimodalBottleneck，L100](../gcnet_missing_m3/meaningful_input_representation.py#L100)；修改 Local 与 memory。
- **实际机制：** Local、Base、有效 Gap 各保留独立流；先做各自私有 Transformer，再进行 2 轮融合。每轮各流拼接同一旧版 4-token bottleneck，分别经过完整 Transformer；所有流提案计算完后才平均 bottleneck，供下一轮回传，最终返回流 token 而非只读瓶颈。
- **为何放在已有读出：** 测试受限的跨角色通信与角色内 head 处理，能否同时保留来源差异、交换共同信息。
- **主要迁移限制：** 这里的流是读出角色，不是原论文音视频序列；Local 单 token 的私有 self-attention 没有流内配对作用。角色提案等权平均可能稀释强证据。bottleneck 每个话语重新生成，不是 OSRAM 新缓存。
- **来源：** [论文](https://arxiv.org/abs/2107.00135) · [作者代码（固定版本）](https://github.com/google-research/scenic/blob/8c113c501c9f700b69899c55a69e65bb46727da6/scenic/projects/mbt/model.py)。

### 11. NetVLAD — `netvlad_residual_encoding`

- **实现/写回：** [meaningful_input_representation.py，_NetVLAD，L136](../gcnet_missing_m3/meaningful_input_representation.py#L136)；修改 Local 与 memory。
- **实际机制：** token L2 归一化 → 8 个可学习 soft assignment → 相对独立可学习中心的残差累积 → 簇内 L2 → 全局 L2；随后用原 assignment 将簇残差描述重新分配给原 token，再零桥写回。assignment 参数与残差中心独立。
- **为何放在已有读出：** 测试相对学习原型的一阶残差统计，是否比简单均值更能表示某个 head 与总体读出的偏离。
- **主要迁移限制：** 分配回原 token 是新增适配器，并非 NetVLAD 原有重构算法；随机初始化代替 k-means，没有地点检索监督、triplet loss 或外部特征库。短集合可能留下闲置 codeword，残差抵消也可能损失信息。
- **来源：** [论文](https://arxiv.org/abs/1511.07247) · [作者代码（固定版本）](https://github.com/Relja/netvlad/blob/652dbe71aa45c691961ddd9f6cf902574e6bdc2f/layerVLADv2.m)。

### 12. Token Merging / ToMe — `tome_merge_reconstruct`

- **实现/写回：** [meaningful_input_representation.py，_TokenMerging，L202](../gcnet_missing_m3/meaningful_input_representation.py#L202)，合并为 [_merge_evidence，L159](../gcnet_missing_m3/meaningful_input_representation.py#L159)；修改 Local 与 memory。
- **实际机制：** 两个 block 均按 attention → 合并 → FFN 执行；用归一化平均 key 做确定性二部匹配，每个源选最佳目的，再选最多 4 条最高分源边。按累计 size 加权合并，后续 attention 加 `log(size)`；保存成员索引并反向展开回原槽位。
- **为何放在已有读出：** 测试显式合并相似读出、保留未合并 minority head 并补偿代表质量，能否为 Flat 提供更有用的聚合上下文。
- **主要迁移限制：** 交替下标只用于算法分区，不代表空间或时间邻接；硬匹配不求导，展开只是复制，不可逆恢复。**Local 被保护不参加合并，但其 Flat 输入仍会被 attention/解码修正。** 无预训练 ViT 或短集合加速保证。
- **来源：** [论文](https://arxiv.org/abs/2210.09461) · [作者代码（固定版本）](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/merge.py)。

### 13. MCAN 编码器—解码器 — `mcan_encoder_decoder`

- **实现/写回：** [meaningful_input_vqa.py，MCANInput，L216](../gcnet_missing_m3/meaningful_input_vqa.py#L216)，核心为 [_MCAEncoder，L68](../gcnet_missing_m3/meaningful_input_vqa.py#L68) 和 [_MCADecoder，L81](../gcnet_missing_m3/meaningful_input_vqa.py#L81)；修改 Local 与 memory。
- **实际机制：** Q 固定为 Base 的 8 个真实 head；V 为 Local 加有效 Gap，即 1/9/17 个 token。6 个 Q self-attention/FFN encoder 后，6 个 V decoder 各执行 self-attention → 受最终 Q 引导的 cross-attention → FFN；保留各子层残差及 source 风格 sample-std 归一化。
- **为何放在已有读出：** 测试共同历史作为多 token 条件，能否引导当前 Local/缺失相关 Gap 的组合；避免把唯一 Local 当作退化的单 token question bank。
- **主要迁移限制：** Base head 不是词，V 不是图像 patch；只迁移 MCA_ED 交互核心，不迁移 AttFlat 或答案头。引导方向不对称，Q 在编码后不再接收 V 的反馈；全观测时 V self-attention 仍为单 token。
- **来源：** [论文](https://arxiv.org/abs/1906.10770) · [作者代码](https://github.com/MILVLG/mcan-vqa)。

### 14. Dense Symmetric Co-Attention / DCN — `dense_coattention`

- **实现/写回：** [meaningful_input_vqa.py，DenseCoattentionInput，L230](../gcnet_missing_m3/meaningful_input_vqa.py#L230)，核心为 [_DenseSymmetricLayer，L95](../gcnet_missing_m3/meaningful_input_vqa.py#L95)；修改 Local 与 memory。
- **实际机制：** 沿用 Base8 与 Local+Gap 双 bank；5 层各给两边追加 3 个学习 null token，用同一配对 affinity 的两个方向归一化。4 个 affinity head 聚合完整 128 维 value 后取平均，再基于同一旧状态同时做两边 `x + ReLU(W[x,read])` 更新；null 不写回。
- **为何放在已有读出：** 测试反复双向对齐是否比单向 Base 引导更适合协调角色分歧，并允许“不去关注”另一侧的选项。
- **主要迁移限制：** dense 指全配对，不是 DenseNet 历史层拼接；这也不只是 MCAN 换深度。null 不代表补全模态，可能在小 bank 占据过多权重；源残差没有额外归一化，不能据名称推断稳定性或效果。
- **来源：** [论文](https://arxiv.org/abs/1804.00775) · [作者代码](https://github.com/cvlab-tohoku/Dense-CoAttention-Network)。

### 15. MAC 控制—读取—写入 — `mac_control_read_write`

- **实现/写回：** [meaningful_input_vqa.py，MACInput，L241](../gcnet_missing_m3/meaningful_input_vqa.py#L241)，核心为 [_MACCell，L125](../gcnet_missing_m3/meaningful_input_vqa.py#L125)；**只修改 Flat 的 Local 输入，Base/Gap 不改写**。
- **实际机制：** Local 投影形成 query；Base8 为 control context，Local+有效 Gap 为 knowledge bank。4 次共享 cell 依次执行 control attention、上一临时状态条件的知识读取、线性 write；各步另有 query 投影。最终将 query 与第 4 步临时状态拼接，通过零桥修正 Local。
- **为何放在已有读出：** 测试同一批已读证据上的多步条件选择，能否给 Flat 的当前输入补充有用的推理上下文。
- **主要迁移限制：** 临时 reasoning memory 每个话语重置，绝非 OSRAM Memory 写入；没有自然语言推理监督。采用论文的基本 recurrent-control/read 配置，区别于作者仓库默认开关；可选 write gate/self-attention 未启用，不声称默认模型复现。
- **来源：** [论文](https://arxiv.org/abs/1803.03067) · [作者代码](https://github.com/stanfordnlp/mac-network)。

### 16. SPDNet — `spdnet_bimap_reeig_logeig`

- **实现/写回：** [meaningful_input_spectral.py，SPDNetInput，L111](../gcnet_missing_m3/meaningful_input_spectral.py#L111)，配合 [OrthogonalBiMap，L86](../gcnet_missing_m3/meaningful_input_spectral.py#L86)；修改 Local 与 memory。
- **实际机制：** 有效 64 维读出描述的中心化协方差加 `1e-4 I`，经过 `64→32→16→8` 三层正交 BiMap；前两层 ReEig 下界分别为 `1e-3/1e-2`，最后 LogEig，展开为 64 维全局特征，再用独立输出坐标桥写回。W 由斜对称矩阵指数乘固定正交基构造；谱反传使用 Loewner 差商处理重复特征值。
- **为何放在已有读出：** 测试跨描述通道的二阶共变关系及分层谱变换，是否能补充原 Flat 对证据分布形状的利用。
- **主要迁移限制：** 协方差来自异质读出，并非原始 SPD 观测；少量 token 导致低秩与 ridge 影响。矩阵指数参数化替代原 Stiefel-SGD 轨迹，固定递增阈值是显式适配；自定义谱梯度仅支持一阶，不能宣称原几何优化过程复现或不确定性校准。
- **来源：** [论文](https://arxiv.org/abs/1608.04233) · [作者代码](https://github.com/zhiwu-huang/SPDNet)。

### 17. Neural Logic Machines — `neural_logic_machine_evidence`

- **实现/写回：** [meaningful_input_logic.py，NeuralLogicInput，L106](../gcnet_missing_m3/meaningful_input_logic.py#L106)，核心为 [LogicLayer，L68](../gcnet_missing_m3/meaningful_input_logic.py#L68)；仅修改 memory，Local 参与条件计算但返回原值。
- **实际机制：** Local 与每个完整 512 维有效 memory role 分别投影到 32 维，形成 2/3/4 个对象，**不是 head-token 图**。初始化一元/二元软谓词；4 层在 arity 0/1/2/3 上同步执行升阶、exists-max/forall-min 降阶、全部变量排列、共享 MLP+sigmoid。最终零元与对应一元特征共同解码到 memory 角色。
- **为何放在已有读出：** 测试多角色关系经显式量词式聚合和跨 arity 传播，是否能提供单次配对聚合之外的任务信息。
- **主要迁移限制：** 学到的是向量软谓词，不是已知逻辑事实；2 个对象时排除重复变量后的三元量化常为空。最终只读 arity 0/1，故末层二元推断 4,656、末层三元推断 6,704、倒数第二层三元推断 6,704，合计 **18,064 个参数没有任务梯度**；该数由源码维度与依赖路径静态核算，不是训练测量。不能称所有推断参数都由任务学习。
- **来源：** [论文](https://arxiv.org/abs/1904.11694) · [作者代码（固定版本）](https://github.com/google/neural-logic-machines/tree/3f8a8966c54d13d2658c77c03793a9a98a283e22)。

### 18. FSPool + FSUnpool — `fspool_fsunpool_evidence`

- **实现/写回：** [meaningful_input_sorting.py，SortingInputAdapter，L94](../gcnet_missing_m3/meaningful_input_sorting.py#L94)，核心为 [SortPoolUnpoolCore，L73](../gcnet_missing_m3/meaningful_input_sorting.py#L73)；仅修改 memory，Local 参与集合但不写回。
- **实际机制：** 64 维逐 token encoder → 每个 feature 的确定性 NeuralSort 软排序（温度 1）→ 20 段分段线性 rank 权重池化 → 分离的 encoder-latent 与 decoder-latent MLP → decoder 专用 rank 权重展开 → 保存的软排序矩阵转置回原 token → 逐 token decoder → 零桥。
- **为何放在已有读出：** 测试按值的相对秩统计及逐 token 回传，是否能捕捉 evidence 集合中极值、尾部或中间位置的信息，而不假设 head 有时间顺序。
- **主要迁移限制：** 逐 feature 排序可能弱化跨 feature 对应关系；软矩阵转置不是真逆，latent 不是无损压缩。没有源重构损失，解码的只是已有读出修正，绝不是缺失模态补全。
- **来源：** [论文](https://arxiv.org/abs/1906.02795) · [作者代码（固定版本）](https://github.com/Cyanogenoid/fspool/tree/a9f93cc774610c6d96c2c3095a1ab16f53abbefb)。

### 19. RAT-SPN 概率电路 — `rat_spn_evidence_circuit`

- **实现/写回：** [meaningful_input_circuit.py，_RATSPNInput，L90](../gcnet_missing_m3/meaningful_input_circuit.py#L90)，核心为 [_GaussianCircuit，L28](../gcnet_missing_m3/meaningful_input_circuit.py#L28)；仅修改 memory。
- **实际机制：** Local 及四角色的 8 个 head 各自投影到 4 个标量，共固定 132 个变量；4 棵固定随机分区树、深度 3，叶部 8 个归一化 Gaussian 分量。不同子 scope 枚举全交叉 product，在 log 域相加；归一化 sum 权重经 logsumexp 层层聚合到 32 个 root 特征，再零桥写回。分区固定保存，不在 forward 重采样。
- **为何放在已有读出：** 测试一个平滑、可分解的层级联合证据表示，能否通过任务损失学习有用的角色/坐标组合。
- **主要迁移限制：** 非激活变量把叶 log-density 设为 0，表示积分掉该投影坐标，**不是把缺失值当作观测零，也不是补全原模态**。没有 NLL/EM 新目标；密度只定义在学习投影上，不保证校准，root 数值还受观测变量数量和尺度影响。
- **来源：** [论文](https://proceedings.mlr.press/v115/peharz20a.html) · [作者代码（固定版本）](https://github.com/cambridge-mlg/RAT-SPN/tree/366e33a6488d8a8d16ba9a2ba512e7f2779afd7b)。

### 20. Provably Powerful Graph Networks / PPGN — `ppgn_pair_composition`

- **实现/写回：** [meaningful_input_pair.py，PairInputAdapter，L75](../gcnet_missing_m3/meaningful_input_pair.py#L75)，核心为 [_PairBlock，L39](../gcnet_missing_m3/meaningful_input_pair.py#L39)；修改 Local 与 memory。
- **实际机制：** 128 维真实 token 的有序端点特征加相等指示，提升为 64 通道二阶 pair tensor。3 个完整 block 各用独立双层 ReLU 分支 f/g，逐通道执行 `P[i,j,c] = Σ_k f(X[i,k,c]) g(X[k,j,c])`，拼接旧 pair 后线性压缩；不是 feature-channel dot product。每层读取自身对角、全局对角 max、全局非对角 max，三层拼成每 token 576 维并零桥解码。
- **为何放在已有读出：** 测试经第三个证据节点组合两段关系，是否能给原 Flat 提供直接 Local—history 配对之外的信息。
- **主要迁移限制：** 二阶状态需要 `O(n²c)` 存储、`O(n³c)` 乘积；未归一化乘法可能放大数值。它不同于 RRN 的节点递归、EGT 的边状态注意力、Sheaf 的节点扩散及 NLM 的 arity/量词链；但新的连续证据 pair 提升和逐槽回读不继承源图分类器的 3-WL 保证。两个 pair 轴是证据身份，不是二维图像空间。
- **来源：** [论文](https://proceedings.neurips.cc/paper/2019/hash/bb04af0f7ecaee4aae62035497da1387-Abstract.html) · [作者代码（固定版本）](https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/blob/4576eff7dc9137c70bdcef724392b82a1642c562/layers/modules.py)。

## 使用本清单时的限制

以上链接用于追溯迁移依据；本地实现是针对既有读出的数学机制适配，不是整篇论文、原 checkpoint、原优化轨迹或原训练目标的复现。候选卡保留作者、版本、许可与去重记录；来源链接不是源码复制授权。尤其不能把 null、symbol、bottleneck、MAC 临时状态或概率电路边缘化描述成新增 OSRAM Memory、真实对象解释或缺失模态恢复。

本文只做文档与源码静态核对，没有启动运行、训练或重新执行测试，也不报告最新任务完成状态。实际数值证据、资源消耗与训练结果应读取独立运行记录；即使 smoke 通过，也不能据此声称性能改善。
