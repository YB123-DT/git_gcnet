# MOSI Shared Post-Graph BiLSTM 设计规格

## 1. 目标

在冻结 wav2vec、DeBERTa、MANet 特征和单模型边界内，检验 GCNet 的
Temporal/Speaker 两个图分支是否因重复的图后 BiLSTM 容量而在小规模 MOSI 上
过拟合。

本轮只回答一个问题：两张关系图保留独立图卷积后，共享图后序列动态是否优于两套
独立 BiLSTM。

用户已要求持续自主执行，不等待逐步确认；本规格据此前的持续授权进入实现。

## 2. 证据与候选比较

- MOSI 训练集只有 52 段对话。
- Fresh deterministic Legacy 在 epoch 45–50 左右达到验证最佳；到 epoch 100，
  训练 W-F1 约为 94%，Val8 已降至约 71%–72%，存在明显过拟合。
- hidden=100 时，两支图后 BiLSTM 各含 2,508,000 个参数，合计接近模型可训练参数
  的一半。
- Temporal-only 和 Speaker-only 都低于双分支，不能删除任一关系图。
- Packed recurrent、availability-conditioned readout、affine conditioning 和 JEPA
  聚合修正均未形成稳定提升。

比较过的方案：

1. **只共享图后 BiLSTM（采用）：** 保留两张图及两套输出 linear，减少重复的序列
   容量，仍允许 Temporal/Speaker 输出使用不同几何。
2. **共享 BiLSTM 与 linear（拒绝）：** 正则更强，但同时消除分支输出差异，首轮
   无法判断收益来自哪一层。
3. **Conversation Availability Context（后备）：** 参数少且直接编码对话缺失率，
   但此前多个小型条件器均失败，其先验弱于当前的过拟合证据。

## 3. 架构

新增开关：

```text
postgraph_sequence_mode = independent | shared-bilstm
```

默认 `independent` 必须保持现有行为。`shared-bilstm` 的数据流为：

```text
Temporal RGCN/GraphConv -> Temporal raw sequence ┐
                                                 ├-> Temporal grufusion
Speaker  RGCN/GraphConv -> Speaker raw sequence  ┘

Temporal recurrent output -> Temporal linear -> ReLU ┐
Speaker  recurrent output -> Speaker  linear -> ReLU ├-> Original addition
```

精确边界：

- Temporal/Speaker 的 `RGCNConv`、`GraphConv` 和 `linear` 保持独立；
- 两路 forward 都调用 `graph_net_temporal.grufusion`；
- `graph_net_speaker.grufusion` 仍按原顺序构造并保留 state-dict keys，但候选模式下
  冻结且不执行；
- 两次共享 BiLSTM 调用仍分别产生 dropout mask；
- 保留现有 Legacy padding、MSE task loss、target-balanced JEPA、Shared readout、
  branch addition、mask schedule 和 checkpoint 选择；
- 不组合 SmoothL1、Packed recurrent 或其他候选。

## 4. 兼容性与参数

所有模块继续按 Legacy 顺序实例化，因此默认模式的参数、state keys、初始化 RNG、
forward 和 strict checkpoint loading 必须精确不变。

在 `shared-bilstm` 下：

- 总 state-dict key 集不变；
- 旧 checkpoint 可 `strict=True` 加载；
- Speaker BiLSTM 参数保留但 `requires_grad=False`；
- 可训练参数精确减少 2,508,000；
- Temporal 共享 BiLSTM 从两路分支同时接收梯度；
- 推理仍执行两张图和两次图后序列 forward，输出接口不变。

## 5. 接口设计

`GraphNetwork.forward()` 新增可选参数：

```python
postgraph_recurrent: nn.Module | None = None
```

为 `None` 时使用自身 `grufusion`；否则仅在该次调用中使用传入的 recurrent。这个
参数不注册新模块，避免 alias state keys 和重复 checkpoint 语义。

`GraphModel`、`MissingM3GraphModel`、`TrainConfig` 和 CLI 逐层传递
`postgraph_sequence_mode`。候选模式只在 Speaker branch forward 时传入 Temporal
branch 的 `grufusion`。

## 6. 错误处理

- 只接受 `independent` 和 `shared-bilstm`；其他值立即抛出 `ValueError`。
- 候选模式必须同时保留两个图分支；若配置 Temporal-only 或 Speaker-only，则拒绝
  启动，避免产生无意义 treatment。
- Runner treatment-count guard 禁止与 readout、JEPA aggregation、Packed 或
  SmoothL1 同时启用。
- 后续候选强制使用 direct deterministic Legacy control 目录；不得静默回退到旧
  hidden/window control。

## 7. 测试

1. 默认与显式 `independent`：输出、参数、state keys、初始化 RNG 精确相同。
2. `shared-bilstm`：两套 graph conv 与 linear 对象仍不同。
3. 两路 loss 都能向 Temporal 共享 BiLSTM 传递有限非零梯度。
4. Speaker BiLSTM 全部冻结、梯度为空；改变其权重不影响候选输出。
5. 改变任一 branch 的 graph conv 或 linear 会改变候选输出。
6. 可训练参数精确减少 2,508,000，总 state keys 不变，strict loading 通过。
7. CPU 与远程 CUDA FP32 forward/backward 有限。
8. Runner config、manifest、metrics 和 resume audit 记录唯一 treatment。

## 8. 实验协议与停止条件

候选使用 seeds 66、67、68，validation-only，复用 fresh deterministic Legacy MSE
controls。配置固定为 hidden=100、window=1/1、LR=5e-4、Legacy padding、Shared
readout、MSE、target-balanced JEPA 和 all-rates-per-batch。

通过条件：Val8 mean delta 至少 +0.40 个百分点、至少 2/3 seeds 为正、high-missing
delta 非负、miss0 delta 不低于 -0.30、最差 seed 不低于 -1.00，且无坍塌。

若失败，则关闭该路线；不尝试 shared-linear、分层共享、冻结比例或与 SmoothL1
组合。下一独立候选才是 Conversation Availability Context。

