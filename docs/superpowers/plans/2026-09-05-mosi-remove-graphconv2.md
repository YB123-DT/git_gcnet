# MOSI Second GraphConv Ablation Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加一个默认关闭的第二层 GraphConv identity 消融，并用当前 MOSI Text-anchor Test-oracle 协议运行五种子。

**架构：** `GraphNetwork` 始终实例化原 `conv2`，但由 `graph_second_layer` 决定调用 `conv2(out, edge_index)` 还是直接保留 `conv1` 输出。该开关透传到两个图分支和训练配置，默认值保持现有模型逐点等价。

**技术栈：** PyTorch、PyTorch Geometric、pytest、远程 V100 实验。

---

### 任务 1：锁定 GraphConv 执行开关

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_modality_jepa/model.py`

- [x] **步骤 1：编写失败测试**

测试默认 `graphconv` 输出与现有手工 `conv1 -> conv2` 路径一致；测试
`identity` 时 forward hook 证明 `conv2` 未调用、`conv1` 获得有限梯度且
`conv2` 梯度为 `None`；非法值抛出 `ValueError`。

- [x] **步骤 2：运行红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest tests/test_missing_m3.py -k graph_second_layer -q
```

预期：因为 `graph_second_layer` 尚不存在而失败。

- [x] **步骤 3：实现最小路径**

在 `GraphNetwork.__init__`、`GraphModel.__init__` 增加默认参数
`graph_second_layer="graphconv"`，校验 `{"graphconv", "identity"}`，并在
forward 中实现：

```python
out = self.conv1(features, edge_index, edge_type)
if self.graph_second_layer == "graphconv":
    out = self.conv2(out, edge_index)
```

两个图分支接收同一开关，`conv2` 始终构造。

- [x] **步骤 4：运行绿灯并提交**

```bash
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest tests/test_missing_m3.py -k graph_second_layer -q
```

### 任务 2：透传 Missing-M3 配置和 CLI

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [x] **步骤 1：编写失败测试**

验证 CLI 默认 `graphconv`、候选 `identity`，`TrainConfig`/保存的 config
记录相同值，并验证 Temporal/Speaker 两个分支均收到 `identity`。

- [x] **步骤 2：运行红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest tests/test_missing_m3.py -k graph_second_layer -q
```

- [x] **步骤 3：实现透传**

在 `MissingM3GraphModel`、`TrainConfig`、argument parser、模型构造和配置保存
路径增加 `graph_second_layer`，默认 `graphconv`。

- [x] **步骤 4：运行相关及完整测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest tests/test_missing_m3.py -q
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest -q
```

### 任务 3：远程五种子 Test-oracle 实验

**文件：**
- 创建：`experiments/missing_m3_mosi_no_graphconv2_test_oracle_20260905/RESULT.md`
- 创建：`experiments/missing_m3_mosi_no_graphconv2_test_oracle_20260905/results/seed_*/`

- [x] **步骤 1：同步并运行**

在现有 MOSI Text-anchor 命令中只增加：

```bash
--graph-second-layer identity --checkpoint-selection test-oracle
```

并运行 seeds 66--70；对照继承 `62208ae` 中的 Text-anchor history。

- [x] **步骤 2：核验结果**

确认五份 100-epoch history、4,000 个有限逐-rate Test W-F1、40 个 NPZ、
无 traceback。逐 seed/rate 独立选择 Test 最大 W-F1 及对应 epoch。

- [x] **步骤 3：形成结论并上传**

报告五种子 rate 均值、SD、相对 Text-anchor 的逐-rate delta 和正向 seed 数；
明确标记 Test-oracle。使用 Lore commit 提交并推送
`feature/missing-m3-target-ple`。
