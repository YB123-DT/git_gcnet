# Complete-Preserving Missing-Pattern FiLM RGCN 设计规格

## 目标与研究理由

GCNet 第一层 `RGCNConv` 对每个 relation 的邻居执行固定均值，无法依据发送节点和接收节点的缺失模式改变特征传播。主候选从非 MSA/MERC 领域的 GNN-FiLM 引入目标节点条件化的逐维调制，同时加入发送节点的缺失模式修正。新增路径以完整模态为原点，因此全完整输入严格恢复官方 GCNet。

该方法不把 relation-aware attention 当作创新。RGAT、DualGATs 和 HGT 已覆盖 source/target content 与 relation-conditioned propagation；可检验的新贡献边界是 ordered source-target missing-pattern conditioning、GCNet 固定关系图上的逐维调制，以及 complete-preserving 约束。

## 锁定模块

将 temporal graph 和 speaker graph 的第一层 `RGCNConv` 同时替换为 `MissingPatternFiLMRGCNConv`。第二层 `GraphConv`、图后 BiLSTM、MatchingAttention、两分支相加、分类器、重建头、损失和拓扑均不改变。

输入为 `x[N,D_in]`、`edge_index[2,E]`、`edge_type[E]`、原始 `node_mask[N,3]`。七种合法 mask 映射为六维 contrast code：A、T、V、AT、AV、TV 分别为一个 one-hot 位置，ATV 为零向量；000 非法。完整状态直接由 `node_mask.bool().all(-1)` 判断，不由零向量反推。

对 relation `r` 的边 `u -> v`：

\[
q_{u,r}=h_uW_r+p_uP_r,
\]

\[
[\Delta\gamma_{v,r},\Delta\beta_{v,r}]
=[h_v\Vert p_v]G_r,
\]

\[
a_{uv}=1-c_uc_v,
\]

\[
m_{u\to v}^{r}
=(1+a_{uv}\Delta\gamma_{v,r})\odot q_{u,r}
+a_{uv}\Delta\beta_{v,r}.
\]

每个 relation 内使用与 PyG `RGCNConv(aggr="mean")` 相同的 `scatter_mean`，然后原样加入 root transformation 和 bias。`P_r` 与 `G_r` 零初始化；因此任意 missing pattern 下初始化前向等于 Original，而当全图完整时新增参数在整个训练过程中始终被 `a=0` 关闭。

## 变体

- `original`：官方 PyG `RGCNConv`。
- `pattern_only`：发送 pattern correction 保留，FiLM generator 只读取目标 pattern，不读取目标内容。
- `full`：上述完整公式。
- `content_film_control`：不读取任何 mask，用目标内容生成 FiLM，并通过零参数填充与 Full 对齐参数量；用于排除“只是参数更多或内容调制”的解释。

Count-only 和三个全局 modality scalar 仅作为后续机制消融，不进入第一轮正式 A/B。

## 数据流与固定 mask

`input_features_mask[0]` 是经过 speaker 选择后的 `[T,B,3]` 可用性。它必须按 `batch_graphify` 的 conversation-major 顺序展平：逐个 batch item 截取 `[:length]` 后拼接，禁止直接 reshape。

固定 mask bank 使用官方 `random_mask` 生成规则，但每个 `missing_rate × mask_seed` 只生成一次，并按 conversation ID 与 utterance index 保存。所有模型变体读取同一 bank，训练期间不重新采样。有效 mask 不包含 000；在 `eta=0.7` 时只包含 A、T、V。

## 验收测试

1. 全完整 forward 与 PyG `RGCNConv` 最大误差 `<1e-6`。
2. 全完整 input、relation weight、root、bias backward 最大误差 `<1e-5`，新增参数梯度严格为零。
3. 七种 pattern 编码准确，000 被拒绝。
4. `eta=0.7` 的 A/T/V 在人工非零参数下产生不同输出。
5. A-only 同质邻域中 Full 仍受 source/target content 调制。
6. 单邻居 relation 的 mean、root、bias 正确。
7. modality mask 与 `batch_graphify` 节点顺序一致。
8. 记录四个变体的参数量并验证控制组匹配约束。
9. CPU/GPU FP32 forward/backward 通过。

## 第一轮实验判据

数据集仅使用 IEMOCAP-6。先做 fold 5、固定 mask bank、Original 对 Full 的 smoke 和短程筛选；通过后执行 8 个 missing rates、5 seeds 的锁定 A/B。主要指标为 weighted F1，同时报告 accuracy、最佳 epoch、最小/最大验证 F1、跨 seed 均值与标准差。坍塌判据必须同时查看预测类别覆盖、多数类占比和 F1 轨迹，不能只凭某个单点 F1 下结论。

## 已知风险

- FiLM 参数量较大，IEMOCAP-6 上可能过拟合；首轮不额外增加网络层。
- 无约束 `1+delta_gamma` 可能变为负值；先按 GNN-FiLM 原机制测试，只有日志显示数值不稳定时才开启有预注册的稳定性修正实验。
- 2026 HGDN 全文尚不可核验，因此不得使用“首次”表述。
