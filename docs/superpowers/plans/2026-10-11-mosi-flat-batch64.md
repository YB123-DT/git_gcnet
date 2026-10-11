# MOSI cfg84 Flat Batch64 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 可复现地启动一个 MOSI Flat seed66、batch64、LR2e-3、150 epoch实验。

**架构：** 复用现有 `run_experiment` 和同种子 cfg84 Flat reference，只通过独立 runner应用三个显式配置差异。运行使用不可变 git archive和原数据 manifest。

**技术栈：** Python、PyTorch、pytest、git archive、tmux、nvidia-smi。

---

### 任务1：锁定配置

**文件：**
- 创建：`experiments/osram_mosi_flat_batch64_20261011/__init__.py`
- 创建：`experiments/osram_mosi_flat_batch64_20261011/run.py`
- 创建：`tests/test_mosi_flat_batch64.py`

- [ ] 先写测试，断言相对 reference 只改变 epochs、batch_size、learning_rate，且64/32的倍率等于2e-3/1e-3的倍率。
- [ ] 运行测试确认 runner 尚不存在而失败。
- [ ] 实现配置验证、训练、恢复、artifact和mask核对。
- [ ] 运行目标测试、py_compile和diff检查。
- [ ] 使用 Lore 格式提交。

### 任务2：密封并启动

**文件：**
- 创建：`experiments/osram_mosi_flat_batch64_20261011/LAUNCH.md`

- [ ] 从实现提交创建不可变 git archive并生成 `SNAPSHOT.json`。
- [ ] 在 biggpu s0环境运行目标测试。
- [ ] 核对GPU0 UUID/显存、数据/reference hash和磁盘。
- [ ] 用持久tmux启动单一训练，确认进程、配置、模型上卡和首轮日志。
- [ ] 写入实际启动证据并使用 Lore 格式提交。

### 任务3：完成后汇总

**文件：**
- 创建：`experiments/osram_mosi_flat_batch64_20261011/RESULT.md`

- [ ] 结合退出状态与20个必要 artifacts确认完成。
- [ ] 汇总每率指标、mean8/high、BEST epoch及Flat150差值。
- [ ] 明确说明这是联合 batch/LR 配方，不解释成纯batch消融。
- [ ] 使用 Lore 格式提交结果。

