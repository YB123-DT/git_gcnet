# MOSI Branch-specific Post-graph BiLSTM Ablation Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现 Temporal-only 与 Speaker-only Post-BiLSTM bypass，并分别运行 MOSI 五种子 Test-oracle。

**架构：** `GraphNetwork` 保留全部参数，通过 branch-level boolean 选择真实 BiLSTM 输出或复制原序列以匹配现有 linear 输入。`GraphModel` 将一个三值消融开关路由到两个图分支。

**技术栈：** PyTorch、PyTorch Geometric、pytest、远程 V100。

---

### 任务 1：TDD 实现分支 bypass

**文件：** `tests/test_missing_m3.py`、`gcnet_modality_jepa/model.py`

- [ ] 编写测试：bypass 时 `grufusion` 不调用、参数梯度为 None，输出等于复制序列后经过同一 linear 的公式；默认路径保持逐点等价。
- [ ] 在 biggpu `s0` 环境运行 `pytest tests/test_missing_m3.py -k postgraph_bilstm_ablation -q`，确认因 API 缺失而失败。
- [ ] 增加 `postgraph_bilstm_enabled`，bypass 公式为 `outputs=torch.cat([outputs, outputs], dim=-1)`，其余 attention/linear/ReLU 路径不变。
- [ ] 重跑聚焦测试并确认通过。

### 任务 2：配置、CLI 与双分支路由

**文件：** `tests/test_missing_m3.py`、`gcnet_missing_m3/model.py`、`gcnet_missing_m3/train_gcnet.py`、`gcnet_modality_jepa/model.py`

- [ ] 测试 `none/temporal/speaker` CLI、默认值、非法 shared-bilstm 组合以及两个分支的布尔状态。
- [ ] 增加 `postgraph_bilstm_ablation` 并透传至 config、模型、保存记录；默认 `none`。
- [ ] 运行 `pytest tests/test_missing_m3.py -q`，预期全部通过。
- [ ] 使用 Lore commit 提交实现。

### 任务 3：两组五种子实验

**文件：** `experiments/missing_m3_mosi_postgraph_bilstm_ablation_test_oracle_20260905/`

- [ ] T-off 使用 `--postgraph-bilstm-ablation temporal`，S-off 使用 `speaker`；其他参数与 commit `62208ae` 完全一致并保留 GraphConv2。
- [ ] 每组运行 seeds 66--70、100 epochs，一张卡一个任务，避开 GPU4。
- [ ] 核验 10 histories、1,000 epochs、8,000 finite scores、80 NPZ、零 traceback。
- [ ] 分别计算 T-off/S-off 对 inherited control 的逐-rate五种子 delta、正向种子数及总体均值，写入 `RESULT.md` 并推送当前 GitHub 分支。
