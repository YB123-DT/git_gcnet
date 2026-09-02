# Current Missing-M3 JEPA Contribution Ablation 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 复用现有 runner 完成当前 IEMOCAP-6 模型的五种子 `jepa_weight=0` 配对消融，并以预注册门槛决定是否扩展 MOSEI。

**架构：** 仅把现有 IEMOCAP runner 的数据集集合与 JEPA 权重暴露为参数，默认值保持原来的双数据集和 `0.1`。训练器与模型不修改。

**技术栈：** Python、pytest、PyTorch、biggpu official GCNet environment。

---

### 任务 1：参数化现有 runner

**文件：**
- 创建：`tests/test_missing_m3_iemocap_current_runner.py`
- 修改：`scripts/run_missing_m3_iemocap_current.py`

- [ ] 编写测试，断言默认构造仍为 IEMOCAP-6/4 共 10 个任务。
- [ ] 编写测试，断言指定 `datasets=("IEMOCAPSix",)`、`jepa_weight=0.0` 只构造五个任务且命令包含唯一权重。
- [ ] 运行测试并确认因缺少新参数而失败。
- [ ] 最小修改 `_build_jobs` 和 CLI，使测试通过。
- [ ] 运行 focused test 与 `git diff --check`。

### 任务 2：远程五种子实验

**文件：**
- 输出：`/data2/yb/remote_experiments/missing_m3_iemocap6_no_jepa_20260902/formal`

- [ ] 同步经过验证的源码到 biggpu development copy。
- [ ] 检查 Python、GPU 0/1/2/3/7 与远程目录；禁止 GPU 4。
- [ ] 先 dry-run，确认恰好五个 IEMOCAP-6、`jepa_weight=0` 任务。
- [ ] 启动五个 100-epoch 任务并等待全部完成。
- [ ] 回传轻量 config、history、metrics、prediction NPZ 与日志。

### 任务 3：配对审计与决策

**文件：**
- 创建：`experiments/missing_m3_iemocap6_jepa_ablation_20260902/EXPERIMENT.md`
- 创建：`experiments/missing_m3_iemocap6_jepa_ablation_20260902/results/SUMMARY.json`

- [ ] 验证 5/5 histories、40/40 predictions、mask hashes 和配置锁。
- [ ] 从 NPZ 独立重算 W-F1，按 seed/rate 与现有 JEPA 结果配对。
- [ ] 计算 miss0、八率、高缺失均值与正向 seed 数。
- [ ] 按设计门槛决定是否启动 MOSEI No-JEPA。
- [ ] 使用 Lore commit 提交代码、文档和轻量结果并推送当前 GitHub 分支。
