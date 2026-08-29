# CMU-MOSI Hidden–Window 扫描实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建并启动固定 `time-attention=False` 的 12配置×3 seed 可恢复扫描。

**架构：** 一个本地可测试的 Python runner 生成不可变任务矩阵，并在 biggpu 上为每个任务启动独立训练进程。任务以 `metrics.json` 为完成标志，以 `status.json` 和 `train.log` 提供恢复与诊断证据。

**技术栈：** Python 3、`subprocess`、pytest、现有 `gcnet_missing_m3.train_gcnet`。

---

### 任务 1：锁定任务矩阵

**文件：**
- 创建：`scripts/run_mosi_hidden_window_sweep.py`
- 创建：`tests/test_mosi_hidden_window_sweep.py`

- [ ] 编写测试，要求 36 个唯一任务、每卡 12 个、每 seed 覆盖 12 个配置。
- [ ] 运行测试并确认因 runner 不存在而失败。
- [ ] 实现 `build_jobs()` 与 `build_command()`，命令固定正式 Slot 配置且不含 `--time-attn`。
- [ ] 运行测试并确认通过。

### 任务 2：实现可恢复执行

**文件：**
- 修改：`scripts/run_mosi_hidden_window_sweep.py`
- 修改：`tests/test_mosi_hidden_window_sweep.py`

- [ ] 测试 dry-run manifest、完成任务跳过和独立日志路径。
- [ ] 实现 manifest、并发子进程、状态文件与非零退出汇总。
- [ ] 运行 runner 单测及现有 Missing-M3 CLI 相关测试。

### 任务 3：远程启动与核验

**文件：**
- 创建：`experiments/missing_m3_mosi_hidden_window_sweep_20260829/EXPERIMENT.md`

- [ ] 同步 runner 到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev`。
- [ ] 在 biggpu dry-run 并验证36任务矩阵。
- [ ] 以后台进程启动 GPU 0/1/2 的正式扫描。
- [ ] 检查 runner、36个子进程、每卡显存和首批日志无即时异常。
- [ ] 将启动证据写入实验账本。
