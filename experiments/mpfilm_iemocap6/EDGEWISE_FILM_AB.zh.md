# Linearized FiLM 与 Faithful Edge-wise FiLM 判别实验

## 结论

在 IEMOCAP-6、fold 5、8 个缺失率和 5 个固定种子的锁定协议下，恢复 GNN-FiLM 的“逐边 ReLU 后再聚合”并未产生稳定提升。完整模态下两种实现逐元素完全一致；在 35 个非完整配对上，Faithful Edge-wise FiLM 为 15 胜、20 负，平均加权 F1 差值仅为 `+0.000391`（约 `+0.04` 个百分点）。

Faithful 在缺失率 0.5 和 0.7 上分别提高 `+1.13` 和 `+0.68` 个百分点，但都只有 3/5 seeds 获胜，95% 置信区间跨 0，Wilcoxon 检验不显著。缺失率 0.1 则 5/5 seeds 下降，平均下降 `1.75` 个百分点。因而逐边 activation 是实现忠实性修正，但当前证据不足以将其认定为 GCNet 的有效优化模块。

## 研究问题

此前的 Missing-Pattern FiLM 将调制后的消息直接做 relation-wise mean：

\[
m_{u\to v}^{r}=
(1+\Delta\gamma_{r,v})\odot
(W_rh_u+P_rp_u)+\Delta\beta_{r,v}.
\]

Microsoft 的 GNN-FiLM 官方实现先对所有逐边调制消息应用 activation，再按目标节点聚合。官方代码注释也明确写出 `Activation only after FiLM modulation`，执行顺序为 `modulated_messages -> activation_fn(all_messages) -> message_aggregation_fn`。参考：[GNN-FiLM 论文](https://proceedings.mlr.press/v119/brockschmidt20a.html)与[固定 commit 的官方实现](https://github.com/microsoft/tf-gnn-samples/blob/ff14b96ad97be3dc0c6829d2ad54a63e10779a94/gnns/gnn_film.py)。

本实验只检验一个假设：此前无稳定增益是否由遗漏逐边 activation 导致。

## 唯一实现差异

### Linearized FiLM

\[
\tilde h_v^r=
\operatorname{mean}_{u\in\mathcal N_r(v)}m_{u\to v}^{r}.
\]

### Faithful Edge-wise FiLM

对于至少一端缺失的 active edge：

\[
\tilde h_v^r=
\operatorname{mean}_{u\in\mathcal N_r(v)}
\operatorname{ReLU}(m_{u\to v}^{r}).
\]

对于 source 和 target 都完整的边不施加 ReLU，保持原消息；当全图完整时直接调用 PyG `RGCNConv.forward`。这使 Faithful 成为 complete-preserving 的 GNN-FiLM 适配，而不是对完整 GCNet 额外插入非线性层。

除此之外，两臂的 source pattern、target pattern/content FiLM、relation 参数、root、bias、第二层 GraphConv 和后续网络完全相同。实现 commit 为 `0c8dd83`。

## 实现前验证

共运行 33 个单元/集成测试，CPU 与 CUDA 全部通过。与本 A/B 直接相关的检查如下：

1. Linearized 与 Faithful 参数量相同，均为 `34,712,766`。
2. 全完整 mask 下，Faithful 与 PyG RGCN 的前向、输入梯度、relation weight 梯度、root 梯度和 bias 梯度逐位相同。
3. 人工构造两条消息 `[-2, 1]`：Linearized 聚合结果为 `-0.5`；Faithful 逐边 ReLU 后聚合结果为 `0.5`，证明 activation 位于聚合之前。
4. temporal 与 speaker 两个第一层实例均正确替换，第二层和后续模块不变。

## 锁定实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | IEMOCAP-6 |
| fold | 5 |
| 缺失率 | 0.0、0.1、0.2、0.3、0.4、0.5、0.6、0.7 |
| seeds | 66、67、68、69、70 |
| 两臂 | `linearized_film`、`faithful_edgewise_film` |
| epoch | 100 |
| batch size | 32 |
| hidden | 200 |
| learning rate | 0.001 |
| dropout | 0.5 |
| mask | 同一 immutable mask bank；逐配对 SHA-256 相同 |
| graph/relation | 相同 canonical edge order 与固定 relation ID |
| 硬件 | 4 × NVIDIA TITAN Xp 12 GB；每卡最多 3 个任务 |
| 计算量 | Linearized 约 5.11 GPU-hours；Faithful 约 5.17 GPU-hours |

结果目录：`/data2/yb/paper/experiments/mpfilm_iemocap6_20260824/edgewise_ab_v1/formal`。

实验共 80/80 成功，0 失败，且训练期间没有修改代码。

## 主要结果

以下均为 5 seeds 的加权 F1 均值 ± 标准差；`Δ` 为 Faithful − Linearized。

| 缺失率 | Linearized | Faithful Edge-wise | Δ | 胜/平/负 | Wilcoxon p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.6204 ± 0.0191 | 0.6204 ± 0.0191 | +0.0000 | 0/5/0 | 1.0000 | 1.0000 |
| 0.1 | 0.6226 ± 0.0178 | 0.6051 ± 0.0097 | **−0.0175** | 0/0/5 | 0.0625 | 0.5000 |
| 0.2 | 0.6011 ± 0.0184 | 0.6000 ± 0.0130 | −0.0011 | 2/0/3 | 0.8125 | 1.0000 |
| 0.3 | 0.5969 ± 0.0154 | 0.5932 ± 0.0219 | −0.0037 | 2/0/3 | 0.6250 | 1.0000 |
| 0.4 | 0.5768 ± 0.0207 | 0.5833 ± 0.0230 | +0.0065 | 3/0/2 | 1.0000 | 1.0000 |
| 0.5 | 0.5648 ± 0.0177 | 0.5761 ± 0.0134 | **+0.0113** | 3/0/2 | 0.6250 | 1.0000 |
| 0.6 | 0.5483 ± 0.0079 | 0.5487 ± 0.0272 | +0.0004 | 2/0/3 | 1.0000 | 1.0000 |
| 0.7 | 0.5548 ± 0.0109 | 0.5616 ± 0.0186 | **+0.0068** | 3/0/2 | 0.6250 | 1.0000 |

逐缺失率的 95% 配对差值置信区间全部跨 0。例如：

- 0.1：`[-0.0367, +0.0017]`；
- 0.5：`[-0.0257, +0.0483]`；
- 0.7：`[-0.0170, +0.0306]`。

对每个 seed 先在七个非完整缺失率上取平均，得到差值：

| seed | 平均 ΔF1 |
|---:|---:|
| 66 | +0.004592 |
| 67 | +0.002376 |
| 68 | −0.012305 |
| 69 | +0.010657 |
| 70 | −0.003365 |

这五个 seed-level 均值的总体结果为 `+0.000391 ± 0.008693`，3 胜、2 负；Wilcoxon `p=1.000`，配对 t-test `p=0.925`。由于每个 seed 在多个缺失率上重复出现，35 个单点不能视为 35 个独立样本；因此总体推断以 seed-level 聚合为主，35 个配对的 15 胜/20负仅作为描述。

## 等价性、复现性与坍塌审计

- 两臂 40 个逐任务 mask SHA-256 全部相同。
- 两臂所有任务参数量完全相同。
- 缺失率 0.0 的 5 个 seeds：best epoch、100-epoch loss history、labels、logits、hidden states 和 feature masks 均逐元素相同。
- 本次 Linearized 的 40 个任务与前一次正式 Full 的对应任务逐元素相同，排除了重跑漂移。
- Faithful 的 40 个任务全部预测到 6 个类别；最大主导预测类别占比为 `0.3432`，没有类别坍塌。

## 解释边界

数据支持以下结论：

1. 先逐边 ReLU 再聚合是对原始 GNN-FiLM 的必要实现修正。
2. 该修正在 0.5 和 0.7 上改善了均值，但 seed 间不稳定。
3. 跨完整缺失协议没有稳定净收益，因此遗漏 activation 不是 Linearized FiLM 失败的充分解释。

数据不支持以下说法：

- “Faithful Edge-wise FiLM 已解决 GCNet 缺失模态问题”；
- “Faithful 在高缺失率显著优于 Linearized”；
- “当前模块可以作为最终论文主模块”。

## 局限性

1. 仅运行 IEMOCAP-6 的 fold 5，不能代表完整 5-fold 泛化。
2. 每个条件只有 5 seeds，统计功效有限；即便 Shapiro–Wilk 未拒绝正态性，也不能据此强证正态。
3. 官方 GCNet 流程中 validation loader 与该 fold 的 test loader 相同，best epoch 实际上在 fold-5 上选择，因此结果偏乐观，不能等同于严格独立测试集评估。
4. 本实验只隔离 activation placement，不回答 pattern encoding、target-only conditioning 或 FiLM 参数规模本身是否合理。

## 判定

本次判别实验已经完成，结果为：**Faithful Edge-wise activation 不是稳定有效的修复。** 按锁定规则，应当停止把“补回逐边 activation”作为最终解决方案；是否开启新的机制路线必须另行立项，不能把本实验局部的 0.5/0.7 均值提升包装成正式成功。
