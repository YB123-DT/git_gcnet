# Publishable Angle

## 研究方向

Complete-Preserving Low-Rank Edge-Conditioned Relational Propagation for Incomplete Conversational Emotion Recognition

## 核心创新点

将非 MSA/MERC 领域 ECC 的逐边动态滤波思想适配到 GCNet 固定 temporal/speaker 关系图：由有序 source/target 缺失模式、relation 与双端当前内容生成低秩滤波残差，并在 ATV-to-ATV 时严格恢复官方 RGCN。

## 研究问题

target-only FiLM 已被正式实验判定为无稳定收益。下一问题是：真正按边区分不同 source、target、relation 和内容的动态滤波，能否在不改变图拓扑和完整模态行为的前提下提高高缺失率鲁棒性。

## 方法概要

- 保留 GCNet 的 relation weight、root、bias 和 relation-wise mean；
- 构造 `[source pattern, target pattern, pattern interaction, relation embedding, source-target content interaction]` edge descriptor；
- 用四个秩 8 动态滤波 basis 形成逐边残差；
- 仅在至少一个端点缺失时激活新增路径；
- 先通过 11-task 晋级门，再决定是否扩展到完整协议。

## 对标基线

- 官方 GCNet `RGCNConv`；
- 已归档 Missing-Pattern Linearized FiLM；
- 已归档 Faithful Edge-wise FiLM。

## 预期贡献

- 给出缺失模式、relation 与内容联合条件化的逐边关系传播算子；
- 给出完整模态严格保持和固定 mask 的可复现实验协议；
- 通过与 target-only FiLM 的失败证据区分节点条件调制和真正边条件滤波。

## 创新声明边界

ECC 动态滤波不是新发明。可检验贡献仅是 complete-preserving missing-pattern GCNet adaptation。当前撞车等级为 Level 2；2026 HGDN 正文不可访问，因此不得声称首创。

## 选择记录

- 候选方向数：3；
- 用户选择：方向 1，CP-ECC/CP-LECC-RGCN；
- 选择理由：它直接修复已确认的 target-only source 不可区分问题，同时比 Full FiLM 显著减少新增参数；
- 弃选方向：CP-GatedGCN（与 RGAT/门控传播撞车风险较高）；CP-PNA-RGCN（优先解决聚合统计，而非当前已定位的逐边条件缺口）。
