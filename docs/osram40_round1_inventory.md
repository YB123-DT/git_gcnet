# Round1：20 个 Local/Base/Gap 读出模块的实现与结果清单

核对时间：2026-10-04 UTC；代码位置按当前工作树（HEAD `b227161cf87bd3efa53572e2f74f313c6c05150e`）逐个检查。范围仅为 [ROUND1.json](../experiments/osram_meaningful20_20261003/ROUND1.json) 中的 20 项，不含第二轮输入变换。论文/源码链接取自已冻结卡片，本次没有重新联网核验，也没有重新训练。当前文件位置不是原训练 commit 的替代证明；原运行以其不可变快照和 provenance 为准。

## 共同接口与证据边界

- `Local` 是当前话语表示；`Base` 与三个按模态区分的 `Gap` 是**已经完成的** OSRAM 历史读取。新增核心不再次查询或写入 Memory，不接触标签，不把临时迭代状态传到下一话语。这里的“重构”仅在模块内部处理已有证据，不是缺失模态补全。
- [MeaningfulReadoutResidual（meaningful_blocks.py:28）](../gcnet_missing_m3/meaningful_blocks.py#L28) 只取真正的前向 `8×64` heads；有前序有效历史才执行，Base 激活，Gap 仅对当前缺失模态激活。padding、首个有效话语及不激活的证据被排除/清零。在至少一种模态可见的协议下，头级方法实际处理 9/17/25 个节点，而非固定把 33 个槽位都当作观测。
- 18 个核心输出 128 维；TabNet 输出 64 维，NODE 输出 192 维。外层是 `Linear(core.output_dim, 1600)` 零初始化桥，加入原 Flat anchor 后才执行原 `emotion_norm`，见 [osram.py:1733](../gcnet_missing_m3/osram.py#L1733)。不能把本轮统称为“所有方法均为 128 维”，也不能与第二轮直接变换输入的插入点混写。
- 头级输入投影也有差异：graph/hypergraph/set_context 使用各 head 独立的 128 维投影；grouping 使用 64 维、按证据类型投影并加入 Local 条件；optimization 使用跨 head 共享的 128 维投影；feature_reasoning 使用固定 579 维槽位向量。它们不是仅更换名字的同一个注意力块。
- 下表“引入理由”和“预期收益”是**针对该结构的设计假设**，不是已定位的性能瓶颈，也不是实验已证实的增益。注意力、路由、稀疏选择、协方差或内部能量均不能自动解释为可靠性、因果归因或校准不确定性。

结果口径：MOSI，seed66，100 epochs；Mean8 为缺失率 `.0–.7` 的 W-F1 宏平均，表中单位 `%`。每个缺失率使用各自 Test-oracle BEST，可能来自不同 epoch，并非一个统一选中 checkpoint 的表现。来源为 [SUMMARY.json](../experiments/osram_meaningful20_20261003/SUMMARY.json) 和 [RESULT.md](../experiments/osram_meaningful20_20261003/RESULT.md)。所有 20 项均记为 complete；只有一个种子，不提供跨种子稳定性或显著性结论。

## 1. 图消息传递：4 项

本组实现文件：[meaningful_blocks_graph.py](../gcnet_missing_m3/meaningful_blocks_graph.py)。除 EGT 的全连接节点对外，图边由 Local 星形连接、同证据类型及同 head 连接构成；共同按 Local/Base/三个 Gap 分组池化。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `rrn_evidence` / RRN | [RRNCore:50](../gcnet_missing_m3/meaningful_blocks_graph.py#L50)；包装 GraphEvidenceBlock:257 | 5 轮共享参数的成对消息 MLP→邻居求和→拼回初始输入→LSTM 节点更新。 | 让 Local、同 head 的 Base/Gap 在一次读出内反复交换条件，而非一次池化。 | 多轮非线性修订或能表达证据组合中的冲突/一致性。 | 无原逐步监督与 bAbI 编码器；图直径小，5 轮不是获得更远历史；LSTM 状态仅限本次前向。 | [论文](https://arxiv.org/abs/1711.08028) / [源码](https://github.com/rasmusbergpalm/recurrent-relational-networks) | 79.583 |
| `egt_evidence` / EGT | [EGTCore:124](../gcnet_missing_m3/meaningful_blocks_graph.py#L124)，EGTLayer:81 | 3 层全局注意力；显式边状态提供 bias/gate，使用动态 centrality；前两层更新节点和边，末层只更新节点。 | 将 Base/Gap 与 Local 的成对关系保留到下一层，而非每次加权后即丢弃。 | 联合修订节点和关系可能更适合依赖上下文的交互。 | 类型/head 关系替代原任务编码；无 SVD 位置编码或结构辅助目标；边状态不是可信度，末层无边更新。 | [论文](https://doi.org/10.1145/3534678.3539296) / [源码](https://github.com/shamim-hussain/egt_pytorch) | 80.583 |
| `residual_gated_graph_evidence` / Residual Gated Graph ConvNet | [GatedGraphCore:172](../gcnet_missing_m3/meaningful_blocks_graph.py#L172)，GatedGraphCell:159 | 3 个双卷积残差块；端点特征生成逐通道 sigmoid gates，自身映射加门控邻居和。 | 在结构限定的 Local/Base/Gap 边上学习传递哪些特征通道。 | 通道级传递选择可能减少无关信息混入。 | 原 BatchNorm 改 LayerNorm；无 gate-sum 归一化，不是后来的边状态式 GatedGCN；未归一化求和仍受度数影响。 | [论文](https://arxiv.org/abs/1711.07553) / [源码](https://github.com/xbresson/spatial_graph_convnets) | 80.606 |
| `pna_evidence` / PNA | [PNACore:246](../gcnet_missing_m3/meaningful_blocks_graph.py#L246)，PNALayer:216 | 4 层、4 towers；mean/std/min/max 与 3 种度数 scaler 的组合，再更新和混合。 | 缺失模态数改变有效节点数与邻居分布；单个均值可能不足以描述这些变化。 | 多统计量可能保留互补分布信息和极值证据。 | 度数参考来自固定七种合法 mask，而非原数据集统计；度数变化有限；std 不等于不确定性，min/max 也会放大离群值。 | [论文](https://arxiv.org/abs/2004.05718) / [源码](https://github.com/lukecavabarrett/pna) | 80.457 |

## 2. 超图处理：4 项

本组实现文件：[meaningful_blocks_hypergraph.py](../gcnet_missing_m3/meaningful_blocks_hypergraph.py)。实际超边为“同 head”及“同证据类型”两类，Local 加入各超边；不是从真实场景观察得到的图结构。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `allset_transformer` / AllSetTransformer | [AllSetTransformer:130](../gcnet_missing_m3/meaningful_blocks_hypergraph.py#L130)，SeedPooling:103 | 两轮节点→超边→节点的 learned-seed PMA，每次 4 heads，再读出 Local/Base/Gap。 | 同 head 跨 Base/Gap、同类型跨 heads 是两种可显式保留的集合关系。 | 双向集合聚合可能表达不止成对的证据组合。 | 构造超边、输入及任务头均为本任务适配；仅 9–25 节点，不能套用原图规模效率结论。 | [论文](https://arxiv.org/abs/2106.13264) / [源码](https://github.com/jianhao2016/AllSet) | 80.140 |
| `ed_hnn` / ED-HNN | [EDHNN:87](../gcnet_missing_m3/meaningful_blocks_hypergraph.py#L87)，EquivariantDiffusion:31 | 3 次共享迭代：求和 `phi`→结合接收节点的 `rho`→超边返回求和→0.1 初始态重启→`psi`。 | 同一组 Base/Gap 信息可以对 Local 与不同证据节点产生不同反馈。 | 接收者条件化的组级消息可能减少统一广播的信息损失。 | 保留 sum 而非 mean，因此受超边大小影响；不是新增能量损失的数值求解器；无原超图任务头。 | [论文](https://arxiv.org/abs/2207.06680) / [源码](https://github.com/Graph-COM/ED-HNN) | 79.610 |
| `hyper_sagnn` / Hyper-SAGNN | [HyperSAGNN:191](../gcnet_missing_m3/meaningful_blocks_hypergraph.py#L191)，StaticDynamicDiscrepancy:142 | 在两类超边内计算 leave-self-out 动态注意力表示与静态表示的平方差，再汇总。 | 用 Local 所在组中“自身表示与其他证据条件化表示的差异”描述证据不一致。 | 差异向量可能补充直接均值聚合缺少的关系特征。 | 删除超边存在性分类器及负边采样；差异不是校准可靠性；无 Gap 时仍比较 Local/Base，不能解释为缺失恢复。 | [论文](https://arxiv.org/abs/1911.02613) / [源码](https://github.com/ma-compbio/Hyper-SAGNN) | 80.310 |
| `sheaf_hypergnn_diag` / Diagonal SheafHyperGNN | [SheafHyperGNN:255](../gcnet_missing_m3/meaningful_blocks_hypergraph.py#L255)，DiagonalSheafLayer:231 | 两层 4×32 fiber 表示；预测节点–超边对角 restriction maps，左右线性变换后按归一化算子传输。 | 异质 Local/Base/Gap 表示不一定应直接在同一坐标下平均。 | 可学习的传输映射可能保留关系差异，而非只平滑节点。 | 固定对角 maps，非一般 sheaf；实际 `P=D^-1/2+A−2 blockdiag(A)` 追随卡片钉住的作者代码；不得套用论文能量下降定理，近零 map 用 epsilon 稳定。 | [论文](https://arxiv.org/abs/2309.17116) / [源码](https://github.com/IuliaDuta/sheaf_HNN) | 80.776 |

## 3. 竞争分组与路由：4 项

本组实现文件：[meaningful_blocks_grouping.py](../gcnet_missing_m3/meaningful_blocks_grouping.py)。共同使用 [LocalConditionalTokenizer:33](../gcnet_missing_m3/meaningful_blocks_grouping.py#L33)：Local 条件加入每种 Base/Gap head token 的 64 维表示，核心产生 128 维汇总；“组”不是物体或真实情感类别。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `capsule_dynamic_routing` / Dynamic Routing Capsules | [DynamicRoutingGrouping:132](../gcnet_missing_m3/meaningful_blocks_grouping.py#L132)，dynamic_routing:117 | token-specific votes→4 个 32 维父胶囊；3 轮父组 softmax 竞争、squash 和 agreement 累积。 | 将 Local 条件化的多头 Base/Gap 按投票一致性组成少量联合表示。 | 一致证据可能集中到不同潜在组而非全部压成一个平均。 | 无原图像编码器/重构损失；4 组和 3 轮固定，vote 不代表 pose；相关错误也可能被一致性放大。 | [论文](https://arxiv.org/abs/1710.09829) / [源码](https://github.com/Sarasra/models/tree/984fbc754943c849c55a57923f4223099a1ff88c/research/capsules) | 80.274 |
| `slot_attention` / Slot Attention | [SlotAttentionGrouping:150](../gcnet_missing_m3/meaningful_blocks_grouping.py#L150) | 4 个 64 维 slots；输入对 slots 竞争并再按输入归一化，3 次 GRU+MLP 细化；mean/max 汇总。 | 用竞争分配区分当前 Local 条件下的 Base/Gap 证据组。 | 有限 slots 或能分离互补证据、缓解冗余聚合。 | 无对象解码器/对象监督；训练用可保存的私有 RNG，评估固定 seed1729 噪声；不宣称对象发现或 slots 自动去冗余。 | [论文](https://arxiv.org/abs/2006.15055) / [源码](https://github.com/google-research/google-research/tree/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/slot_attention) | 80.144 |
| `otke` / Optimal Transport Kernel Embedding | [OTKernelGrouping:249](../gcnet_missing_m3/meaningful_blocks_grouping.py#L249)，sinkhorn_plan:213 | 非线性归一化特征、4 个 learned supports，log-domain Sinkhorn 30 步，双边质量约束后加权汇总。 | Base/Gap head 数较多；受容量约束的分配提供不同于自由 softmax 的聚合偏置。 | 平衡 support 使用可能避免所有证据集中到一个代表。 | 均匀容量也可能强迫弱证据进入某组；有限步只近似满足边缘分布；无缺失重构或最优情感匹配保证。 | [论文](https://arxiv.org/abs/2006.12065) / [源码](https://github.com/claying/OTK) | 80.075 |
| `capsule_variational_bayes` / VB Capsule Routing | [VariationalBayesGrouping:306](../gcnet_missing_m3/meaningful_blocks_grouping.py#L306)，vb_posterior:272 | 4×4 matrix votes；3 轮完整 Gaussian-Wishart/Dirichlet 后验及协方差感知 responsibility，汇总均值和 activation。 | 用投票分布而非单一内积描述 Local 条件下 Base/Gap heads 的组内差异。 | 协方差感知路由或能区别集中与分散的证据集合。 | matrix 不是物体姿态；entropy activation/归一化有声明的适配；任务监督不保证贝叶斯校准，Cholesky 路由更昂贵。 | [论文](https://doi.org/10.1609/aaai.v34i04.5785) / [源码](https://github.com/fabio-deep/Variational-Capsule-Routing) | 80.090 |

## 4. 集合与上下文读出：3 项

本组实现文件：[meaningful_blocks_set_context.py](../gcnet_missing_m3/meaningful_blocks_set_context.py)，先压紧有效 Local/head tokens，再执行各自的完整处理链。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `perceiver_io` / Perceiver IO | [PerceiverIOReadout:102](../gcnet_missing_m3/meaningful_blocks_set_context.py#L102) | 8 个 latent slots cross-attend 输入→3 层 latent self-attention→由 Local 生成的单查询解码。 | 先联合压缩 Base/Gap 与 Local，再显式按当前 Local 查询联合结果。 | Local 条件化解码可能更有选择地提取历史信息。 | 减小宽度/深度/latent 数；Local 查询属于本任务适配；仅 9–25 tokens，不能以大输入可扩展性作为已验证优势。 | [论文](https://arxiv.org/abs/2107.14795) / [源码](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver) | 80.818 |
| `dgcnn_dynamic_edgeconv` / DGCNN | [DGCNNReadout:153](../gcnet_missing_m3/meaningful_blocks_set_context.py#L153)，_EdgeConv:138 | 4 层重新构建特征空间 kNN，k=4（允许自邻居），聚合 `[邻居−中心,中心]`；多尺度拼接后 max/mean 池化。 | 不固定 Base/Gap 必须如何连接，按当前变换后的证据特征重建邻域。 | 内容相关邻域可能捕捉预设 head/type 图未覆盖的组合。 | 原 xyz 换 learned evidence；BN 换 LN；不保证异质 token 距离有意义，top-k 邻居切换也不连续。 | [论文](https://arxiv.org/abs/1801.07829) / [源码](https://github.com/WangYueFt/dgcnn) | 79.913 |
| `graph_multiset_transformer` / GMT | [GraphMultisetReadout:231](../gcnet_missing_m3/meaningful_blocks_set_context.py#L231)，_GMTMAB:197 | 两层 GCN 多尺度表示→4-seed 图感知池化→seed 间注意力→单 seed 汇总。 | 在 Local 星形和同 head 跨角色图上，先传播再进行分层集合压缩。 | 图结构约束的 key/value 与多 seed 汇总或能保留局部和全局证据。 | 图由任务构造，可能过平滑；按作者实现采用投影 query residual 与总宽度缩放；无图同构/单射或图重构保证。 | [论文](https://arxiv.org/abs/2102.11533) / [源码](https://github.com/JinheonBaek/GMT) | 80.074 |

## 5. 优化式表示：3 项

本组实现文件：[meaningful_blocks_optimization.py](../gcnet_missing_m3/meaningful_blocks_optimization.py)。Hamburger/CRATE 汇总变换前后 token 差值；EA 则直接输出优化得到的 128 维临时聚合量。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `hamburger_nmf_full` / Hamburger NMF | [HamburgerBlock:37](../gcnet_missing_m3/meaningful_blocks_optimization.py#L37) | 两个线性“面包”包住 rank4 非负矩阵分解：6 次无梯度乘性更新，再一次带梯度系数细化与重构。 | 对相关的 Local/Base/Gap 头级证据施加共享低秩结构。 | 提取共享低秩成分可能削减冗余或噪声。 | 固定初始化 basis；截断反传，不是全迭代求导；非负/低秩偏置可能压掉少数但关键的情感证据；无缺失特征补全。 | [论文](https://openreview.net/forum?id=1FvkSpWosOl) / [源码](https://github.com/Gsunshine/Enjoy-Hamburger) | 80.506 |
| `crate_mssa_ista_full` / CRATE | [CRATEBlock:97](../gcnet_missing_m3/meaningful_blocks_optimization.py#L97)，CRATEIteration:69 | 两个完整 MSSA+ISTA 块：共享投影形成子空间交互，随后用 learned dictionary 做 shrinkage；ISTA 替代 FFN，不再加第二个普通残差。 | 联合处理 Base/Gap 与 Local 后，再约束表示的支持集，而非只调最终权重。 | 压缩加稀疏化可能保留更集中的互补特征。 | 2 层 128 维、role/head 输入替代图像 patches；仅原任务损失，不证明 MOSI 上真的最小化 rate 目标；内部仍含注意力。 | [论文](https://arxiv.org/abs/2306.01129) / [源码](https://github.com/Ma-Lab-Berkeley/CRATE) | 80.430 |
| `equilibrium_aggregation` / EA | [EquilibriumBlock:120](../gcnet_missing_m3/meaningful_blocks_optimization.py#L120) | learned 非负势能加正 L2 项及基数缩放；每次从零开始执行 10 步 Nesterov，训练对完整内循环求导。 | 每条 Local/Base/Gap 证据对聚合量的影响可随当前临时聚合量变化。 | 非线性联合优化式汇总可能比固定一次聚合表达更丰富的相互依赖。 | 非凸且仅 10 步，不是精确 equilibrium；删除原梯度范数辅助损失；评估仅为内部求导临时启用梯度；“源码”是出版商补充材料中的作者 listings，不是可直接执行的已验证仓库。 | [论文](https://proceedings.mlr.press/v180/bartunov22a.html) / [作者代码 listings](https://proceedings.mlr.press/v180/bartunov22a/bartunov22a-supp.pdf) | 80.210 |

## 6. 固定槽位特征推理：2 项

本组实现文件：[meaningful_blocks_feature_reasoning.py](../gcnet_missing_m3/meaningful_blocks_feature_reasoning.py)。共同 `FixedSlots:45` 形成 `Local64 + Base128 + GapA128 + GapT128 + GapV128 + availability3 = 579`；不激活 Gap 的坐标被严格排除，而非用一个可选的零值观测替代。

| ID / 名称 | 实际类与行号 | 实际迁移机制 | 为什么引入 Local/Base/Gap | 假设的收益（未证实） | 适配与损失的原方法条件 | 论文 / 源码卡片链接 | Mean8 |
|---|---|---|---|---|---|---|---:|
| `tabnet` / TabNet | [TabNetEvidence:75](../gcnet_missing_m3/meaningful_blocks_feature_reasoning.py#L75) | 一次 warmup 后 3 次 sparsemax 选择；共享/分步 GLU，decision/attention 分流，更新使用 prior 并累加 64 维 decision。 | 让 Local、Base、各 Gap 的固定特征槽按前一步结果分阶段参与决策。 | 带使用预算的顺序选择可能提取不同阶段的互补信息。 | 输入是学习的隐变量而非原始表格列；GhostBN/BN 改 LN，无稀疏辅助损失及自监督预训练；mask 不是因果解释；输出为 64 维。 | [论文](https://arxiv.org/abs/1908.07442) / [源码](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/tabnet/tabnet_model.py) | 79.485 |
| `node` / NODE | [NODEEvidence:162](../gcnet_missing_m3/meaningful_blocks_feature_reasoning.py#L162)，ObliviousTrees:112 | 3 个密集连接树层，各 32 棵深度4树；entmax1.5 特征选择、entmoid 分支概率、完整路径乘积与二维叶响应，拼接成 192 维。 | 用特征阈值的合取路径组合 Local 与各历史槽，区别于连续加权混合。 | 分段决策与后续树层可能表达条件性交互。 | 无源量化预处理、QHAdam 或 checkpoint averaging；首次正常训练的有效历史行初始化阈值/温度；按样本屏蔽缺失槽属于适配，不宣称可解释情感规则。 | [论文](https://arxiv.org/abs/1909.06312) / [源码](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/odst.py) | 79.984 |

## 本轮实测结论与不能推出的结论

原 Flat 的精确 Mean8 为 `81.06809539495711%`；本轮最佳 Perceiver IO 为 `80.81795857267966%`，差 `−0.25013682227745` 个百分点。20/20 项均低于 Flat；按预设八档均值门槛，本轮 `promotions=[]`，未扩展 seed67/68。上述三位小数表仅用于展示，排序和门槛使用未舍入值。

这支持的结论仅是：**这 20 种具体迁移实现，在该单种子固定协议的内部筛选中未胜出**。它不能证明原论文方法无效，不能证明某个 Local/Base/Gap 瓶颈确实存在或已被解决，也不能将本轮最佳候选包装成正式泛化提升。模块内的注意力图、routing、稀疏度、内部能量或残差范数，只能检查计算是否发生；单独都不是情感预测改进的证据。

本次核对直接读取本地实现、20 份结果指标/配置以及已保存汇总；没有重新审计远端完整 checkpoint。20/20 completion、100 轮及各 8 份 BEST/预测的历史远端核验，引用上述 RESULT/SUMMARY 中已有记录。原 Flat 历史训练 commit 未记录，不用后来 commit 冒充。
