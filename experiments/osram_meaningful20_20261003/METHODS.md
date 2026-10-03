# 第一轮20个完整方法的来源与机制

INTERNAL DIAGNOSTIC ONLY

这是资料与接入设计清单，不是性能结果。20项均来自非MSA/MERC工作；表中接入Local/Base/Gap的方案属于本项目适配，原论文不保证其MOSI效果。拒绝项不计入20，不启动已撤回的基础算子队列。

| # | 固定方法ID与论文 | 保留的完整核心 | 详细证据 |
|---|---|---|---|
| 1 | `rrn_evidence`：[Recurrent Relational Networks](https://arxiv.org/abs/1711.08028) | 5轮共享关系消息与LSTM状态更新，反复注入原节点输入 | [来源、实际代码与适配差异](graph.json) |
| 2 | `egt_evidence`：[Global Self-Attention as a Replacement for Graph Convolution](https://doi.org/10.1145/3534678.3539296) | 节点与边双状态同时更新，3层完整边增强Transformer | [来源、实际代码与适配差异](graph.json) |
| 3 | `residual_gated_graph_evidence`：[Residual Gated Graph ConvNets](https://arxiv.org/abs/1711.07553) | 3个双卷积残差单元，端点条件化的邻居信息传递 | [来源、实际代码与适配差异](graph.json) |
| 4 | `pna_evidence`：[Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) | 4层多塔消息传递，统计量×度数缩放联合聚合 | [来源、实际代码与适配差异](graph.json) |
| 5 | `allset_transformer`：[You are AllSet: A Multiset Learning Framework for Hypergraph Neural Networks](https://arxiv.org/abs/2106.13264) | 2轮完整节点→超边→节点的学习集合聚合 | [来源、实际代码与适配差异](hypergraph.json) |
| 6 | `ed_hnn`：[Equivariant Hypergraph Diffusion Neural Operators](https://arxiv.org/abs/2207.06680) | 3轮接收者条件化超边消息与原始状态重启 | [来源、实际代码与适配差异](hypergraph.json) |
| 7 | `hyper_sagnn`：[Hyper-SAGNN: a self-attention based graph neural network for hypergraphs](https://arxiv.org/abs/1911.02613) | 每条超边内独立静态/动态表示与逐维差异 | [来源、实际代码与适配差异](hypergraph.json) |
| 8 | `sheaf_hypergnn_diag`：[Sheaf Hypergraph Networks](https://arxiv.org/abs/2309.17116) | 2层逐关联映射、归一化运输和纤维空间扩散 | [来源、实际代码与适配差异](hypergraph.json) |
| 9 | `capsule_dynamic_routing`：[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) | 4个胶囊、3轮累积一致性路由 | [来源、实际代码与适配差异](grouping.json) |
| 10 | `slot_attention`：[Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055) | 4个临时槽、3轮竞争分配及GRU细化 | [来源、实际代码与适配差异](grouping.json) |
| 11 | `otke`：[A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention](https://arxiv.org/abs/2006.12065) | 4个学习参考支持点、30轮双边缘约束Sinkhorn | [来源、实际代码与适配差异](grouping.json) |
| 12 | `capsule_variational_bayes`：[Capsule Routing via Variational Bayes](https://doi.org/10.1609/aaai.v34i04.5785) | 矩阵投票、全协方差后验、3轮责任度细化 | [来源、实际代码与适配差异](grouping.json) |
| 13 | `perceiver_io`：[Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) | 输入编码→3层潜在交互→Local条件解码 | [来源、实际代码与适配差异](set_context.json) |
| 14 | `dgcnn_dynamic_edgeconv`：[Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) | 4阶段重新建特征近邻图、EdgeConv及多尺度汇聚 | [来源、实际代码与适配差异](set_context.json) |
| 15 | `graph_multiset_transformer`：[Accurate Learning of Graph Representations with Graph Multiset Pooling](https://arxiv.org/abs/2102.11533) | 图编码→图生成K/V池化→摘要交互→最终池化 | [来源、实际代码与适配差异](set_context.json) |
| 16 | `hamburger_nmf_full`：[Is Attention Better Than Matrix Decomposition?](https://openreview.net/forum?id=1FvkSpWosOl) | 完整投影→交替NMF推断→重构→残差路径 | [来源、实际代码与适配差异](optimization.json) |
| 17 | `crate_mssa_ista_full`：[White-Box Transformers via Sparse Rate Reduction](https://arxiv.org/abs/2306.01129) | 2层绑定子空间投影MSSA与字典ISTA组合 | [来源、实际代码与适配差异](optimization.json) |
| 18 | `equilibrium_aggregation`：[Equilibrium Aggregation: Encoding Sets via Optimization](https://proceedings.mlr.press/v180/bartunov22a.html) | 可学习集合能量与10步Nesterov聚合状态更新 | [来源、实际代码与适配差异](optimization.json) |
| 19 | `tabnet`：[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) | 3阶段稀疏特征选择、使用预算与决策表示累加 | [来源、实际代码与适配差异](feature_reasoning.json) |
| 20 | `node`：[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) | 3层密集连接可微树、完整分裂路径与叶响应 | [来源、实际代码与适配差异](feature_reasoning.json) |

## 共同边界

原Flat主路径、OSRAM读写、Query、Key/Value、mask schedule、任务头、任务损失全部保留。新方法仅消费已经得到的Local和已应用读出消融的Base/Gap，新增零初始化输出投影，在原Flat归一化前加入残差。无历史与padding跳过；inactive Gap严格排除。不伪造空间坐标，不把head当作历史utterance。

真实forward512维按原8×64值头拆分。各论文需要的输入适配不同：graph/hypergraph/set采用128维类型头token，grouping采用64维Local条件token，optimization采用共享头投影，TabNet/NODE使用固定579维特征槽。所有差异逐项记录，不能把这一轮视为严格控制容量的机制消融。

只有完成来源/规格审查、源码独立实现、核心算子数值核验、mask/梯度/零初始化兼容性测试后，方法才能进入训练队列。论文核心经过适配，不称为原论文完整复现。

## 来源与限制

Equilibrium Aggregation的作者代码是出版社补充PDF中的JAX Listings，不是已验证可执行Git仓库；PDF SHA固定，并显式记录列表遗漏。部分仓库许可证不明，只作数学与行为参照，不复制代码。多个方法需把跨样本BatchNorm改成逐样本归一化或省略，这些偏离均在卡片中保留。去掉原论文辅助任务时，不声称保留其全部训练保证。

## 评判与循环

每轮完整报告20项，不只挑最好结果。先同协议seed66、100 epochs、8 rates；比较精确同seed Flat，8-rate mean为主，高缺失另报。内部Test-oracle per-rate BEST与反复测试集筛选均有选优偏差，后续3seed复核仍不能当作正式validation选模结果。没有通过正确性/资源验证的条目不得填入伪造分数；记录失败并修复或补充合格替代，保留所有尝试。
