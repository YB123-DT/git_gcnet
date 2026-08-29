# MOSI Branch-Specific Graph-Message Calibration 设计规格

## 1. 根因与边界

Shared Post-Graph BiLSTM 的五种子结果呈现强收缩：candidate 跨 seed SD 约
`0.461` point，control 为 `1.626` points；control 越强，hard-sharing delta 越差，
相关约 `-0.971`。同时它在 miss-0 提升 `0.569` point，却在 `0.5/0.6/0.7`
分别下降约 `1.10/1.12/1.61` points。

这支持如下诊断：两张图在高缺失下需要保留不同的 sequence dynamics，不能共享
BiLSTM；但进入各自 BiLSTM 的 GraphConv message 可能因可见模态减少而发生幅度与
通道分布漂移。下一候选只校准这段 graph message，不共享或删除任何分支。

## 2. 唯一模块

新增开关：

```text
graph_message_calibration = none | branch-layernorm-residual
```

对 Temporal/Speaker 分支各自第二层 GraphConv 输出：

```text
m_b = GraphConv_b(...)
n_b = LayerNorm_no_affine(m_b)
a_b = tanh(alpha_b)
m'_b = m_b + a_b * (n_b - m_b)
```

其中 `alpha_b` 为每个分支独立的 `[D_g]` 参数，零初始化。hidden100 正式配置中
`D_g=50`，两支总计只增加 100 个参数。

精确插入位置：`gcnet_modality_jepa/model.py::GraphNetwork.forward()` 中 `conv2`
之后、`cat([features, out])` 之前。只校准 graph-derived message，原 node feature、
两套 RGCN/GraphConv、两套图后 BiLSTM、branch linear 与最终 addition 全部保留。

## 3. 初始化与兼容

- 默认 `none` 不实例化新参数，state keys、RNG、forward 与旧 checkpoint 保持原样；
- treatment 的 `alpha_b=0`，初始 `m'_b=m_b`；
- `LayerNorm` 不含 affine 参数或 running statistics；
- `tanh(alpha_b)` 将逐通道校准限制在有界 residual 路径；
- Temporal 与 Speaker 的 `alpha` 不共享。

该模块不是 availability readout：它作用在图消息进入 sequence encoder之前；也不是
Shared BiLSTM：它保留两支的独立参数与职责。它不调用 predictor、不恢复模态、不改
推理接口。

## 4. 锁定实验

只改变 `graph_message_calibration`；恢复 Uniform JEPA、MSE、Legacy recurrent、
independent postgraph BiLSTMs 与 shared readout。MOSI hidden100/window1、seeds
66--68、validation-only，继承 direct deterministic Legacy controls。

沿用三种子 gate：overall `>=+0.40` point、`>=2/3` seeds 正、high-missing 非负、
miss-0 `>=-0.30` point、worst seed `>=-1.0` point且无坍塌。通过才补 69/70；
五种子确认前不读取 test。

## 5. 失败边界

若失败，不扫描 LayerNorm/RMSNorm、alpha 初始化、branch-only 校准或与已失败候选组合。
这将表明 frozen-feature 下游的 graph-message normalization 也不足以形成稳定增益，
并构成关闭本轮 graph/sequence 边界微调路线的证据。
