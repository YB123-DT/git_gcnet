# CP-LECC-RGCN 设计规格

## 研究结论与目标

此前 Missing-Pattern FiLM 的正式实验已经排除两个简单解释：Linearized FiLM 在 35 个非完整配对上平均下降，恢复 GNN-FiLM 的逐边 activation 后也只有 `+0.000391` 平均 F1 差值和 15 胜 20 负。因此下一候选不再修改 FiLM activation，而是替换其 target-only 调制假设。

本设计检验：当每条 `source -> target` 边根据有序缺失模式、relation 和双端当前内容生成独立的低秩滤波修正时，能否在完整输入严格恢复官方 RGCN 的前提下改善 GCNet 的缺失模态传播。

方法工作名为 **Complete-Preserving Low-Rank Edge-Conditioned RGCN（CP-LECC-RGCN）**。其论文锚点是 Simonovsky 与 Komodakis 的 [Edge-Conditioned Convolution（CVPR 2017）](https://openaccess.thecvf.com/content_cvpr_2017/html/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.html)及[作者官方实现](https://github.com/mys007/ecc)。原始 ECC 由 edge label 直接生成完整滤波矩阵，并对入邻居做 mean；CP-LECC 保留“逐边动态滤波”核心，但保留 GCNet 的静态 relation weight、root、bias、relation-wise mean，并使用 complete-preserving 低秩残差。因此论文中必须称为 **ECC-inspired GCNet adaptation**，不得称为原 ECC 的忠实复现。

## 锁定替换范围

仅替换 `GraphNetwork` 中 temporal 与 speaker 两支的第一层 `RGCNConv`。以下内容不变：

- `batch_graphify` 拓扑、edge order 和 relation ID；
- 第二层 `GraphConv`；
- 图后 BiLSTM 与 `MatchingAttention`；
- temporal/speaker 分支相加；
- 分类器与 reconstruction head；
- classification/reconstruction loss；
- optimizer、学习率、dropout、epoch 和 mask bank；
- 不增加 edge attention、softmax、学习邻接、额外图层或额外损失。

输入保持为：

- `x: Tensor[N,D_in]`；
- `edge_index: LongTensor[2,E]`，第 0 行 source、第 1 行 target；
- `edge_type: LongTensor[E]`；
- `node_mask: Tensor[N,3]`。

节点 mask 继续使用 A、T、V、AT、AV、TV 六维 one-hot contrast code，ATV 为零向量，000 非法。mask 必须按 `batch_graphify` 的 conversation-major 节点顺序展平。

## 精确消息公式

对 relation `r` 中的边 `j -> i`，基础消息保持：

\[
b_{ji}^{r}=h_jW_r.
\]

### 逐边描述符

内容投影维度锁定为 `D_c=16`：

\[
c_{ji}=(h_iU_i)\odot(h_jU_j),
\quad U_i,U_j\in\mathbb R^{D_{in}\times 16}.
\]

relation embedding 维度锁定为 8：

\[
e_r=E_{rel}[r]\in\mathbb R^8.
\]

逐边描述符为：

\[
z_{ji}=
[p_i\Vert p_j\Vert(p_i\odot p_j)\Vert e_r\Vert c_{ji}]
\in\mathbb R^{42}.
\]

这显式区分有序 pattern pair，并使同一 target、同一 relation 下的不同 source 仍可因 `h_j` 不同而产生不同滤波器。

### 系数生成器

滤波器 basis 数锁定为 `K=4`。系数生成器为单隐藏层 MLP：

\[
q_{ji}=\operatorname{ReLU}(z_{ji}A+a),
\quad A\in\mathbb R^{42\times32},
\]

\[
\alpha_{ji}=\tanh(q_{ji}C+c)\in\mathbb R^4.
\]

隐藏层宽度固定为 32，不使用 normalization 或 dropout。

### 低秩滤波器 basis

每个 basis 的矩阵秩锁定为 `D_b=8`：

\[
B_k=L_kR_k,
\quad
L_k\in\mathbb R^{D_{in}\times8},
\quad
R_k\in\mathbb R^{8\times D_{out}}.
\]

逐边动态修正为：

\[
\Delta m_{ji}=
\sum_{k=1}^{4}
\alpha_{ji,k}(h_jL_k)R_k.
\]

完整保护开关为：

\[
\delta_{ji}=1-\mathbb I[m_i=ATV\land m_j=ATV].
\]

最终消息：

\[
m_{j\to i}^{r}=b_{ji}^{r}+\delta_{ji}\Delta m_{ji}.
\]

本算子内部不增加逐边 activation。原 ECC 的滤波消息也是先乘动态矩阵再 mean；ReLU 位于 filter-generator 隐藏层或完整 ECC 网络的卷积后模块，而不是逐边消息公式中。

### 聚合与输出

基础输出必须直接调用 `super().forward(x, edge_index, edge_type)`，而不是重新手写 RGCN 基础消息。动态分支只聚合 `delta * Delta m`：每个 relation 独立 mean，各 relation 修正求和，再与 PyG 基础输出相加：

\[
h_i^{base}=\operatorname{PyG\text{-}RGCN}(h, E, r),
\]

\[
h_i'=h_i^{base}+
\sum_r
\operatorname{mean}_{j\in\mathcal N_r(i)}
\delta_{ji}\Delta m_{ji}.
\]

这样 correction 为零时，即使图中存在 missing pattern，也能逐位恢复 PyG 基础输出。当全部节点为 ATV 时，forward 进一步使用快速路径直接返回 `RGCNConv.forward`，不构造动态描述符。

## 参数预算

锁定实验中 `D_in=400`、`D_out=100`。

每支新增：

- 双端内容投影：`2 * 400 * 16 = 12,800`；
- 四个秩 8 basis：`4 * (400*8 + 8*100) = 16,000`；
- MLP：`42*32 + 32 + 32*4 + 4 = 1,508`；
- relation embedding：temporal 为 `3*8=24`，speaker 为 `4*8=32`。

因此：

| 分支 | 新增参数 |
|---|---:|
| temporal | 30,332 |
| speaker | 30,340 |
| 合计 | **60,672** |

总模型参数应为 `34,200,838`。新增量是 Full FiLM `572,600` 的约 10.60%。

拒绝 8,248 参数的门控因式分解版本：它生成的是瓶颈 gated message，而不是 ECC 的动态滤波器 basis，无法干净回答逐边 edge-conditioned filter 是否有效。

## 初始化与 RNG 约束

必须避免同时将系数和 basis 置零，否则乘积两侧梯度均为零。

锁定初始化为：

- `U_i`、`U_j`、relation embedding、MLP 隐藏层、`L_k`、`R_k`：Glorot；
- MLP 最后输出层 weight 与 bias：严格为零；
- RGCN `weight/root/bias`：沿用 PyG 初始化。

这样任意 missing setting 的初始输出逐位等于 Original，同时最后输出层第一轮必须获得非零梯度。

新增随机参数初始化必须保存并恢复 PyTorch 全局 CPU RNG state，使随后构造的 `GraphConv`、BiLSTM 和分类器与 Original 使用相同初值。禁止使用固定且跨 seed 相同的私有初始化，因为那会削弱 seed 对新增分支的控制。

## 变体与命名

第一轮只允许：

- `original`：PyG `RGCNConv`；
- `cp_lecc`：本规格完整公式。

在晋级前不实现：

- pattern-only ECC；
- content-free ECC；
- parameter-matched control；
- 更多 basis、不同 rank 或不同 MLP 宽度；
- edge attention、bilinear attention、动态图；
- JEPA 或额外 reconstruction/contrastive loss。

## 测试规格

实现必须先完成以下测试：

1. **Complete forward parity**：将新增参数人工设为非零，全 ATV 时与 PyG RGCN 逐位一致。
2. **Complete backward parity**：input、relation weight、root、bias 梯度逐位一致；新增参数梯度为 `None` 或严格零。
3. **RNG parity**：构造 CP-LECC 不推进全局 RNG；随后构造的下游层与 Original 参数逐位一致。
4. **Trainable zero residual**：missing 输入在初始化时与 Original 相同，但 coefficient output layer 梯度非零且有限。
5. **Ordered pattern pair**：人工参数下 `A -> T` 与 `T -> A` 输出不同。
6. **Source/target content**：分别只改变 `h_j` 与 `h_i`，动态消息均改变。
7. **Relation condition**：固定内容与 mask，只改变 relation，动态消息改变。
8. **Mixed complete edge**：complete-to-complete 边关闭修正，任一端缺失时开启。
9. **Homogeneous pattern neighborhood**：邻居均 A-only 时，不同 source content 仍产生不同消息。
10. **Aggregation semantics**：同 relation 使用 mean，不同 relation 分别 mean 后求和；root 与 bias 正确。
11. **Pattern/node alignment**：七种 pattern、000 拒绝和 conversation-major 节点顺序正确。
12. **Parameter count**：精确验证 30,332、30,340、60,672 和总参数 34,200,838。
13. **Device/dtype**：CPU/CUDA FP32 forward/backward 均通过。
14. **Input validation**：拒绝非法 mask、edge shape、edge type shape 和越界 relation。

## 第一阶段晋级实验

第一阶段只新增 11 个 CP-LECC 任务，不重跑已逐元素验证可复现的 Original 和 Full：

1. `missing_rate=0.0, seed=66, fold=5`：1 个 complete audit；
2. `missing_rate in {0.5,0.7}`；
3. `seed in {66,67,68,69,70}`；
4. fold 5、100 epochs：10 个 missing jobs。

所有任务继续使用同一 immutable mask bank、IEMOCAP-6、4 张 TITAN Xp、每卡最多 3 个任务及已锁定训练参数。

complete audit 必须在 best epoch、100-epoch loss history、labels、logits、hidden states 和 feature masks 上与 Original 逐元素一致。

主要分析先对每个 seed 的 0.5 与 0.7 差值取平均，形成 5 个 seed-level 配对，不能把 10 个 rate-seed 点当作 10 个独立样本。

CP-LECC 只有同时满足下列条件才晋级完整 8 rates：

1. 0.5 和 0.7 相对 Original 的平均 weighted F1 均不下降；
2. seed-level 平均差值至少为 `+0.005`；
3. 至少 4/5 seeds 的两率平均差值为正；
4. seed-level 平均表现同时优于已归档 Full FiLM；
5. 所有任务预测覆盖 6 类，无坍塌；
6. mask hash 与比较臂逐配对一致。

任一条件失败即停止，不调整 rank、basis 数、学习率或其他机制来追逐结果。

## 创新边界与撞车风险

当前审计结论为 **Level 2 强近邻，Level 3 未证实**：

- MMIN、MissModal、UMDF、MPLMM 等处理缺失 MSA/MER，但主要使用重建、蒸馏、prompt 或表征约束；
- SDR-GNN、GSDNet、MGAFR 接近“缺失 MERC + 图传播”，但可访问论文或代码中未发现同时以 `(m_i,m_j,r,h_i,h_j)` 生成逐边动态 relation filter；
- ECC 的逐边动态滤波本身不是新贡献；
- 可主张空间仅为 missing-pattern/content/relation-conditioned、complete-preserving 的 GCNet 适配；
- 2026 年 HGDN 正文不可访问，是正式创新声明前必须保留的未决风险。

因此只能表述为“在已公开且可核验的资料中未发现完全同构方法”，禁止使用“首个”或“首次”。

## 失败判定

CP-LECC 若未通过 11-task 晋级门，支持的结论是：逐边低秩 edge-conditioned filter 仍不足以稳定改善当前 GCNet；不能通过继续堆叠 attention、动态图或额外 loss 将本次失败包装为成功。失败结果必须写入实验过程 Markdown。
