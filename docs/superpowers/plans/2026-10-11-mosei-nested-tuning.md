# MOSEI Nested GNN 小范围调参实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 注册并在 biggpu GPU2/3 上启动六个固定的 MOSEI Nested GNN seed66 容量配置。

**架构：** 复用 `NestedSweepGNN` 和现有 MOSEI cfg84 Flat reference，只新增组合注册与一个范围锁定的实验调度器。每个训练子进程使用同一不可变 git-archive source、同一数据 manifest 和独立输出目录。

**技术栈：** Python、PyTorch、pytest、git archive、tmux、nvidia-smi。

---

### 任务 1：锁定配置注册

**文件：**
- 修改：`gcnet_missing_m3/meaningful_new40_registry.py`
- 创建：`tests/test_mosei_nested_tuning.py`

- [ ] 编写测试，断言六个方法分别映射到设计表中的 `dim/depth/mlp_hidden/groups`，且原版 Nested seeded state_dict 不变。
- [ ] 运行 `pytest -q tests/test_mosei_nested_tuning.py`，确认新增方法尚未注册而失败。
- [ ] 只在 `NESTED_SWEEP` 中增加六个固定配置，不修改 `NestedSweepGNN.forward`。
- [ ] 重跑测试并运行 `pytest -q tests/test_nested_sweep.py tests/test_mosei_nested_tuning.py`；预期全部通过。
- [ ] 使用 Lore 格式提交配置和测试。

### 任务 2：实现范围锁定的 MOSEI runner

**文件：**
- 创建：`experiments/osram_mosei_nested_tuning_20261011/__init__.py`
- 创建：`experiments/osram_mosei_nested_tuning_20261011/run.py`
- 修改：`tests/test_mosei_nested_tuning.py`

- [ ] 增加 runner 测试：任务集合必须恰好六项、全部 seed66；GPU 只允许2/3且每卡三项；配置相对 same-seed Flat 只能改变 `osram_meaningful_block`。
- [ ] 运行目标测试确认 runner 尚不存在而失败。
- [ ] 实现训练子命令：验证 manifest/reference hash、固定100epochs、运行训练、核对24个 rate artifacts、mask hash和 provenance。
- [ ] 实现 dispatch 子命令：GPU2/3 UUID 实时核对，各启动三项，独立日志/目录，禁止已有状态重复提交。
- [ ] 运行 `pytest -q tests/test_mosei_nested_tuning.py` 和 `python -m py_compile experiments/osram_mosei_nested_tuning_20261011/run.py`。
- [ ] 使用 Lore 格式提交 runner 和测试。

### 任务 3：密封、部署并启动

**文件：**
- 创建：`experiments/osram_mosei_nested_tuning_20261011/LAUNCH.md`

- [ ] 用 `git archive HEAD` 创建远程不可变 source，并运行 `experiments/osram_nested_sweep_20261005/seal.py --commit <HEAD>` 生成 `SNAPSHOT.json`。
- [ ] 从现有 MOSEI 三种子 manifest 提取并核验 seed66 数据/reference hashes；不得重新生成特征。
- [ ] 在 biggpu 查询 GPU2/3 UUID、现有进程、空闲显存和磁盘；不终止任何现有进程。
- [ ] 用持久 tmux 启动 dispatcher，核对六个训练 PID、GPU UUID、首个日志和 provenance 状态。
- [ ] 将实际 commit、服务器、GPU UUID、PID、命令、目录和启动检查写入 `LAUNCH.md` 并提交。

### 任务 4：完成后汇总

**文件：**
- 创建：`experiments/osram_mosei_nested_tuning_20261011/RESULT.md`

- [ ] 只有当每个进程退出且 expected artifacts/provenance 验证通过后才标记完成。
- [ ] 汇总每率 ACC/W-F1、mean8、high-missing、参数量和 BEST epoch，并与已有 Flat/Nested seed66 配对比较。
- [ ] 将结果明确标记为 `INTERNAL DIAGNOSTIC ONLY`，保留失败项，不自动启动多种子。
- [ ] 运行汇总一致性测试并使用 Lore 格式提交结果。

