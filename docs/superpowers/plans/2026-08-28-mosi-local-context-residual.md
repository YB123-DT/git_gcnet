# MOSI Local-Context Residual Fusion 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Slot Missing-M3 增加零初始化的 representation-level local modality residual，并以 seed 66 判断是否值得扩展五种子。

**架构：** 三个 Student slot 与 availability 经过小型 MLP产生 GCNet hidden residual；residual 仅进入唯一 sentiment head，M3 predictor 仍读取原 graph hidden。默认关闭并保持旧模型兼容。

**技术栈：** Python、PyTorch、pytest、GCNet、远程 V100。

**执行状态：** 已完成。seed 66 的 `miss=0` W-F1 为 84.40，低于 87.0 扩展门槛，因此按计划停止，未运行 seeds 67–70。完整证据见 `experiments/missing_m3_mosi_local_context_20260828/EXPERIMENT.md`。

---

### 任务 1：TDD 锁定零初始化等价

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`

- [x] 先写测试：`LocalContextResidualFusion` 尚不存在时 import 红灯。
- [x] 测试 local residual `[L,B,Dh]`、padding 零、缺失值无泄漏。
- [x] 构造同 seed Slot base/local，要求 shared state、初始 logits、hidden 和 predictions 精确一致。
- [x] 运行 targeted tests 确认红灯。
- [x] 实现最小模块、末层全零，并让 predictor 使用 `graph_hidden`、classifier 使用 `graph_hidden+residual`。
- [x] 运行 targeted tests 确认绿灯。

### 任务 2：CLI、梯度与回归测试

**文件：**
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 修改：`tests/test_missing_m3.py`

- [x] 先写 CLI/config 和非 Slot 拒绝测试并确认失败。
- [x] 添加 `--local-context-residual`、hidden/dropout 参数与 TrainConfig 路由。
- [x] 验证 local、Student、GCNet、predictor gradients 有限非零。
- [x] 运行完整 `tests/test_missing_m3.py`，要求全部通过。

### 任务 3：远程 smoke 与 seed 66 判别

**文件：**
- 创建：`experiments/missing_m3_mosi_local_context_20260828/EXPERIMENT.md`

- [x] 精确同步三个修改文件到 biggpu 对应路径。
- [x] 运行 1-epoch MOSI GPU smoke，确认 forward/backward/EMA/checkpoint。
- [x] 运行唯一 seed 66、100 epochs、八 rate test。
- [x] 若 miss0 <87.0，记录失败并停止；不得扩五种子。
- [ ] 若 miss0 ≥87.0，补 seeds 67–70并判断五种子 88 门槛。

### 任务 4：审计、审查与推送

**文件：**
- 更新：实验报告与轻量结果目录

- [x] 重算全部 NPZ 指标、检查 paired mask hashes 与 checkpoint SHA256。
- [x] 请求独立代码/结果审查并修复所有 Critical/High/Medium。
- [x] 运行完整 tests 与 `git diff --check`。
- [x] Lore commit、推送 GitHub并核对远程 SHA。
