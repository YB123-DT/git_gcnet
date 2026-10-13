# IEMOCAP Correct-Mask Nested Completion 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在正确 GCNet missing-mask 协议下完成 IEMOCAP-4/6 原版 Nested GNN 三种子五折实验。

**架构：** 扩展现有 correct-mask IEMOCAP runner，使 Flat/Nested 只由一个显式 variant 控制；新增只调度 Nested 的双卡队列，避免重复 Flat 与 MELD。所有训练语义继续由现有 `run_experiment` 提供。

**技术栈：** Python、PyTorch、pytest、tmux、biggpu V100。

---

### 任务 1：锁定配对配置

**文件：**
- 创建：`tests/test_gcnet_mask_nested_runner.py`
- 修改：`experiments/osram_gcnet_random_mask_20261013/run.py`

- [ ] 测试 Flat/Nested 配置除 `osram_meaningful_block` 外完全一致。
- [ ] 测试两个配置都固定 `gcnet-random-mask-v1`、100 epochs 和对应数据集。
- [ ] 运行 `pytest -q tests/test_gcnet_mask_nested_runner.py`，先确认缺少接口而失败。
- [ ] 为 `_iemocap_config` 和 CLI 增加 `model_variant`，Nested 映射到 `nested_gnn_rooted_evidence`。
- [ ] 重跑测试并确认通过。

### 任务 2：建立有限双卡队列

**文件：**
- 创建：`experiments/osram_gcnet_random_mask_20261013/dispatch_nested.py`
- 修改：`tests/test_gcnet_mask_nested_runner.py`

- [ ] 测试队列恰好包含30个唯一任务，4类全部绑定GPU0、6类全部绑定GPU1。
- [ ] 实现每卡最多3个并发、持久状态记录、独立日志和失败终态。
- [ ] 运行目标测试、`git diff --check` 和 Python 编译检查。

### 任务 3：提交、部署与启动

**文件：**
- 修改：`experiments/osram_gcnet_random_mask_20261013/RESULT.md`

- [ ] 在 RESULT 中记录启动范围、commit、远程根目录和内部诊断口径。
- [ ] 使用 Lore commit 提交并推送 `feature/ecerc-features`。
- [ ] biggpu 拉取同一 commit，确认 GPU0/1 UUID 与空闲状态。
- [ ] 在 tmux 中启动队列并核对30个任务、首批6个PID、有效配置和显存。

