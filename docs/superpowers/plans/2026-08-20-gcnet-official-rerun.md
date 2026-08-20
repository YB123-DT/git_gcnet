# GCNet 官方协议四数据集重跑实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加可审计的 GCNet 官方评估模式，并在固定环境中启动四数据集、两方法、8 缺失率、10 种子的 640-run 重跑。

**架构：** 训练器通过 `evaluation_protocol` 分派 official/strict 生命周期；manifest 根据协议校验 split 和 test 调用次数；一个可恢复调度器生成完整矩阵并限制每卡并发。现有 strict 协议保持兼容。

**技术栈：** Python 3.10、PyTorch、unittest/pytest、CUDA、SSH、JSON manifest。

---

### 任务 1：锁定官方数据与生命周期行为

**文件：**
- 创建：`tests/test_official_evaluation_protocol.py`
- 修改：`gcnet_modality_jepa/train_gcnet.py`

- [ ] 编写测试：parser 默认 `official`；IEMOCAP official 的 validation/test 使用相同 held-out 索引；每 epoch 各调用一次 validation/test；最终 test 来自 validation 最优 epoch。
- [ ] 运行 `pytest -q tests/test_official_evaluation_protocol.py`，确认因协议参数和生命周期尚未实现而失败。
- [ ] 添加 `--evaluation-protocol {official,strict}`，实现 official loader 和 official epoch loop，保留 strict 分支。
- [ ] 重跑目标测试，确认通过。

### 任务 2：让掩码调度和 manifest 感知协议

**文件：**
- 修改：`gcnet_modality_jepa/mask_schedule.py`
- 修改：`gcnet_modality_jepa/run_manifest.py`
- 修改：`gcnet_modality_jepa/train_gcnet.py`
- 测试：`tests/test_mask_schedule.py`
- 测试：`tests/test_run_manifest.py`
- 测试：`tests/test_run_manifest_integration.py`

- [ ] 先增加失败测试：official evaluation mask 随 epoch 确定性变化；official manifest 允许 IEMOCAP validation/test 重合；official test 调用数等于实际 epoch，strict 仍为 1。
- [ ] 运行上述三个测试文件并确认预期失败。
- [ ] 最小实现 `freeze_evaluation` 和协议化 manifest 校验/audit。
- [ ] 重跑测试并确认通过。

### 任务 3：实现 640-run 可恢复调度器

**文件：**
- 创建：`scripts/run_official_missing_sweep.py`
- 创建：`tests/test_official_sweep.py`

- [ ] 先测试矩阵恰好 640 个唯一任务、IEMOCAP 均为 fold5、GPU4 被拒绝、已完成任务会跳过。
- [ ] 运行测试确认调度器尚不存在而失败。
- [ ] 实现任务生成、每卡并发限制、日志、状态文件、续跑和 pair audit。
- [ ] 重跑测试确认通过。

### 任务 4：本地与官方环境验证

**文件：**
- 修改：`docs/experiments/UNIFIED_PROTOCOL.md`
- 创建：`docs/experiments/OFFICIAL_RERUN_20260820.md`

- [ ] 在固定解释器运行目标测试和完整测试套件。
- [ ] 记录 Python/Torch/CUDA/PyG 版本及测试结果。
- [ ] 用 Lore 格式提交代码，且不纳入已有无关脏文件。

### 任务 5：远程 paired smoke

**文件：**
- 远程输出：`/data2/yb/experiments/gcnet_official_smoke_20260820/`

- [ ] 同步已提交代码到 biggpu，验证 GPU4 未使用。
- [ ] 四数据集各运行 baseline/JEPA、fold5（IEMOCAP）、2 epochs 的真实数据 smoke。
- [ ] 审计 4 对 manifest；任何失败先修复并重跑。

### 任务 6：启动并监控正式队列

**文件：**
- 远程输出：`/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/`

- [ ] 生成并保存 640-task manifest，检查数据集/方法/rate/seed/fold 计数。
- [ ] 在 GPU `0,1,2,3,5` 启动，每卡最多 3 个并发，使用固定 official 环境和共享只读数据缓存。
- [ ] 检查首批日志、GPU 显存、失败计数和完成标记；确认异常时自动停止新任务并保留续跑状态。
- [ ] 把启动命令、PID、状态查询命令和首批证据写入实验记录。
