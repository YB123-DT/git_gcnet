# Target-Private Gradient Causality A/B 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 CMU-MOSI 上完成 `target_private_rank=0` 与 `32` 的五种子严格配对实验，判断强制共享近正交目标梯度是否损害下游表现。

**架构：** 复用已经测试的 Target-Private MMoE 实现，只新增一个可恢复的实验 runner。runner 为两臂生成完全相同的训练参数和 seed/mask 配对，隔离输出目录，完成后按 rate 汇总 treatment-control delta。

**技术栈：** Python 3.10、PyTorch、现有 `gcnet_missing_m3` trainer、pytest、远程 V100/A100 GPU 环境。

---

### 任务 1：锁定 runner 的配对合同

**文件：**
- 创建：`tests/test_mosi_target_private_runner.py`
- 创建：`scripts/run_mosi_target_private_ab.py`

- [ ] **步骤 1：编写失败的 job-matrix 测试**

测试 `_build_jobs(...)` 精确生成 10 个任务，arms 为 `shared`/`target-private`，seeds 为 66–70；同 seed 的命令除 `--target-private-rank` 与输出目录外完全一致，并排除 GPU 4。

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest -q tests/test_mosi_target_private_runner.py`

预期：因 runner 模块不存在而失败。

- [ ] **步骤 3：实现最小 job builder**

runner 使用不可变 `Job` dataclass，并固定：

```python
ARMS = {"shared": 0, "target-private": 32}
SEEDS = (66, 67, 68, 69, 70)
RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
```

训练命令调用 `python -m gcnet_missing_m3.train_gcnet`，固定当前 slot、dual-gate、stratified 配置，只由 arm 写入对应 rank。

- [ ] **步骤 4：验证 job builder**

运行：`pytest -q tests/test_mosi_target_private_runner.py`

预期：PASS。

### 任务 2：实现继承、调度与汇总

**文件：**
- 修改：`scripts/run_mosi_target_private_ab.py`
- 修改：`tests/test_mosi_target_private_runner.py`

- [ ] **步骤 1：测试兼容性检查**

构造临时 `config.json`、`metrics.json`、`best.pt`，验证只有 arm、seed 和全部锁定配置一致时任务才可继承；半写 JSON 和缺失文件必须返回未完成。

- [ ] **步骤 2：实现原子状态与 bounded GPU queue**

每个任务使用独立目录；状态先写 `.tmp` 再 replace。每张 GPU 同时一个进程，GPU 列表由 CLI 提供且拒绝 4。异常子进程被记录为 failed，不阻塞其他队列。

- [ ] **步骤 3：实现 paired summary**

读取两臂 `metrics.json` 的八个 test rates，输出：每 seed/rate delta、rate mean、positive seed count、non-zero macro delta 和门槛 verdict。

- [ ] **步骤 4：运行 runner 测试**

运行：`pytest -q tests/test_mosi_target_private_runner.py tests/test_missing_m3.py -k 'target_private or target_private_runner'`

预期：所有相关测试 PASS。

### 任务 3：远程正式执行与分析

**文件：**
- 创建：`experiments/missing_m3_target_private_ab_20260904/EXPERIMENT.md`
- 生成：`experiments/missing_m3_target_private_ab_20260904/paired_summary.json`

- [ ] **步骤 1：同步并启动**

使用远程官方 Python 环境，GPU 0、1、2、3、7 执行 10 个任务；不使用 GPU 4。

- [ ] **步骤 2：检查训练合同**

核对 10 个 config、mask provenance、selection split 与 checkpoint 均完整，且同 seed 两臂除 rank/参数量外一致。

- [ ] **步骤 3：生成判决**

按书面规格计算每 rate 五种子配对结果。若 Treatment 未通过门槛，明确记录“没有因果证据”，不得只挑选改善 rate。

- [ ] **步骤 4：提交结果**

运行 `git diff --check` 和相关 pytest，使用 Lore commit 协议提交并推送 GitHub 分支。

