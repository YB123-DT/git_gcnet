# MOSEI Nested GNN 小范围调参设计

## 目标与边界

在现有 CMU-MOSEI cfg84 no-JEPA Flat/Nested 协议上，仅改变 Nested GNN 的内部容量，检验 MOSEI 是否更适合更小、更浅或稍宽的局部图分支。保持 ObservedSetEncoder、OSRAM Memory 读写、Flat、任务头、损失、cyclic missing、100 epochs、seed66 和 per-rate Test-oracle 评测不变。

## 固定实验矩阵

原版 `dim=64, depth=3, mlp_hidden=64, groups=8` 已有结果，不重跑。新增六组：

| 方法 | dim | depth | mlp_hidden | groups |
|---|---:|---:|---:|---:|
| `mosei_nested_dim32` | 32 | 3 | 32 | 8 |
| `mosei_nested_depth2` | 64 | 2 | 64 | 8 |
| `mosei_nested_dim32_depth2` | 32 | 2 | 32 | 8 |
| `mosei_nested_mlp128` | 64 | 3 | 128 | 8 |
| `mosei_nested_groups4` | 64 | 3 | 64 | 4 |
| `mosei_nested_dim32_depth2_mlp64` | 32 | 2 | 64 | 8 |

这些配置只改变图节点维度、GIN 深度、GIN MLP 宽度或证据节点分组；OSRAM 仍为 8 个真实 Memory heads、每 head 64 维。`groups=4` 只在 Nested tokenizer 中将相邻真实 head 拼成四组，不改变 Memory 本身。

## 执行与资源

服务器固定为 `biggpu`。宿主机 GPU2、GPU3 各并行三组；启动时通过实际 UUID 绑定，禁止 GPU4。每组使用独立输出、日志、配置快照、checkpoint 和 provenance。若任一卡现有任务导致显存不足，不终止现有进程，等待容量释放。

## 判定与报告

报告每率 ACC/W-F1、8-rate mean、high-missing mean、参数量、BEST epoch 和运行状态。第一轮只使用 seed66 做内部筛选；不把 Test-oracle 数值作为正式论文结果。仅当候选稳定超过现有 Nested 和 Flat 的主要汇总指标时，另行决定是否补 seed67/68。

## 正确性要求

- 新配置注册必须精确映射到表中四个结构参数。
- 默认 `nested_gnn_rooted_evidence` 的构造与输出不变。
- 所有新配置保持 residual=True、zero decoder=True、root marker、same-head edge 和逐层汇总。
- 调度器只允许上述六项、seed66、MOSEI、GPU2/3，并防止重复启动。

