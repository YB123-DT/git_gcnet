# Flat History Recency Metadata 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 cfg84 原始 Flat 增加严格过去的六维 modality coverage/recency 元信息，并完成 seed66、150 epoch 内部筛选。

**架构：** 独立纯函数从 availability 与 umask 计算 `R_t`；仅在开关开启时扩展 Flat adapter 输入六维。CHM 与任务头保持不变，实验使用独立 runner 和不可变远程源码。

**技术栈：** Python、PyTorch、pytest、现有 MOSI 训练/续训 runner、SSH biggpu。

---

### 任务 1：严格因果元信息

**文件：**
- 创建：`gcnet_missing_m3/history_recency.py`
- 创建：`tests/test_history_recency.py`

- [ ] **步骤 1：编写失败测试**：覆盖首句、上一句距离、间隔距离、当前观测不泄漏、padding 不更新状态。
- [ ] **步骤 2：验证红灯**：运行 `pytest -q tests/test_history_recency.py`，预期因模块缺失失败。
- [ ] **步骤 3：最小实现**：按时间循环，在输出当前 `R_t` 后才用当前 availability 更新 `seen/last_seen`。
- [ ] **步骤 4：验证绿灯**：同一 pytest 命令全部通过。
- [ ] **步骤 5：提交**：仅提交纯函数与测试。

### 任务 2：原 Flat 可选集成

**文件：**
- 修改：`gcnet_missing_m3/osram.py`
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 修改：`tests/test_history_recency.py`

- [ ] **步骤 1：编写失败测试**：验证默认关闭输出/state_dict 不变、开启后 adapter 输入增加六维、竞争模块被拒绝、梯度有限。
- [ ] **步骤 2：验证红灯**：运行目标测试，预期开关参数缺失。
- [ ] **步骤 3：最小集成**：新增 `osram_history_recency`，仅扩展原 Flat adapter 并拼接元信息。
- [ ] **步骤 4：回归验证**：运行目标测试及现有 counterfactual readout 测试。
- [ ] **步骤 5：提交**：提交配置与模型集成。

### 任务 3：可复现实验入口与启动

**文件：**
- 创建：`experiments/osram_flat_history_recency_20261011/run.py`
- 创建：`experiments/osram_flat_history_recency_20261011/RESULT.md`
- 创建：`tests/test_history_recency_runner.py`

- [ ] **步骤 1：编写失败测试**：固定配置 delta 只能是 `osram_history_recency: false -> true`，并验证 100@1e-3 + 50@1e-4 日程。
- [ ] **步骤 2：实现 runner**：复用 complete150 runner，保存 manifest、provenance、smoke 与汇总。
- [ ] **步骤 3：本地及远程验证**：运行测试和真实 MOSI 三步 forward/backward smoke。
- [ ] **步骤 4：Git 提交**：使用 Lore commit 记录验证与边界。
- [ ] **步骤 5：正式启动**：建立不可变源码快照，在 biggpu 健康 GPU0 同卡追加 seed66 任务并记录 PID、日志和显存。

