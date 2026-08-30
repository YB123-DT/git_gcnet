# MOSI Fixed-Rate Missing-M3 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为现有 Slot Missing-M3 增加一个 rate 训练、同 rate 选模和同 rate 测试的正式协议，并运行 CMU-MOSI 8 rates × 5 seeds。

**架构：** 在现有训练器末端追加 fixed-rate 配置，不改变模型或 loss；通过独立 runner 生成、恢复和审计 40 个任务。旧 `cyclic`/`all` 路径保持不变。

**技术栈：** Python 3.8、PyTorch 1.8、pytest、subprocess、SSH biggpu、Tesla V100。

---

### 任务 1：锁定 fixed-rate 训练合同

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

增加测试，要求 `TrainConfig(train_rate_mode="fixed", fixed_missing_rate=0.5)` 的两个
batch 都只准备 `0.5` view，并验证非法 rate、fixed 缺 rate、非 fixed 携带 rate 均报错。

- [ ] **步骤 2：运行红灯**

运行：

```bash
/home/yangbin/miniconda3/envs/gcnet-official/bin/python -m pytest \
  tests/test_missing_m3.py -k 'fixed_rate' -q
```

预期：因缺少 `fixed_missing_rate`/`fixed` choice 而失败。

- [ ] **步骤 3：最小实现**

在 `TrainConfig` 末尾追加：

```python
fixed_missing_rate: float | None = None
```

增加 `_fixed_missing_rate(config)` 校验函数，并在 `train_epoch()` 中令 fixed 模式每个
batch 只生成：

```python
rate_views = ((rate, _prepare_view(data, schedules[rate], epoch, dimensions)),)
```

- [ ] **步骤 4：运行绿灯**

重复步骤 2，预期 fixed-rate focused tests 全部通过。

- [ ] **步骤 5：提交**

提交训练合同，commit 使用 Lore trailers，记录 positional compatibility 与未改模型。

### 任务 2：同率验证、选模与测试

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

增加最小 fake loader/model 集成测试，要求 fixed `0.5` 时 history 只含 validation `0.5`、
checkpoint score 等于该 rate W-F1、metrics/test/NPZ 也只含 `0.5`；旧 all 模式仍包含八率。

- [ ] **步骤 2：运行红灯**

运行 fixed selection 测试，预期当前八率 validation/test 行为导致失败。

- [ ] **步骤 3：最小实现**

新增 `_protocol_rates(config)`：fixed 返回单元素 tuple，其他模式返回 `MISSING_RATES`。
`run_experiment()` 的 validation/test 循环和 selection mean 只遍历该 tuple；metrics 增加
`train_missing_rate` 与 `selection_missing_rates`。

- [ ] **步骤 4：运行绿灯和既有回归**

运行：

```bash
/home/yangbin/miniconda3/envs/gcnet-official/bin/python -m pytest \
  tests/test_missing_m3.py -q
```

预期：全部通过。

- [ ] **步骤 5：提交**

提交同率 lifecycle 变更，并记录旧 mixed-rate 行为已回归验证。

### 任务 3：可恢复 40-task runner

**文件：**
- 创建：`scripts/run_mosi_fixed_rate.py`
- 创建：`tests/test_mosi_fixed_rate_runner.py`
- 创建：`experiments/missing_m3_mosi_fixed_rate_20260830/EXPERIMENT.md`

- [ ] **步骤 1：编写 runner 失败测试**

测试默认矩阵恰为 40 个唯一 `(rate, seed)`，命令包含 `--train-rate-mode fixed` 和匹配的
`--train-missing-rate`，只使用 GPU 0/1/2，不包含 Original 命令；完成检查拒绝缺少
100 epochs、错误 config 或多个 test rate 的目录。

- [ ] **步骤 2：运行红灯**

运行：

```bash
/home/yangbin/miniconda3/envs/gcnet-official/bin/python -m pytest \
  tests/test_mosi_fixed_rate_runner.py -q
```

预期：runner 模块不存在而失败。

- [ ] **步骤 3：实现 runner**

复用现有 runner 的原子 JSON、进程回收和 manifest 方式，固定当前正式 Slot 配置；提供
`--dry-run`、`--jobs-per-gpu`、`--python`、`--output-root` 和 resume。

- [ ] **步骤 4：运行绿灯与 dry-run**

运行 runner tests，并确认 dry-run 输出任务数 40、GPU 仅 0/1/2。

- [ ] **步骤 5：提交**

提交 runner、测试和实验登记。

### 任务 4：远程正式执行与归档

**文件：**
- 生成：`experiments/missing_m3_mosi_fixed_rate_20260830/results/SUMMARY.json`
- 生成：`experiments/missing_m3_mosi_fixed_rate_20260830/results/SUMMARY.md`
- 生成：`experiments/missing_m3_mosi_fixed_rate_20260830/results/PROVENANCE.json`
- 修改：`experiments/missing_m3_mosi_fixed_rate_20260830/EXPERIMENT.md`

- [ ] **步骤 1：同步并做一次官方环境验证**

在 biggpu 使用 `/data2/yb/reproduction_envs/gcnet-official/bin/python` 运行 focused tests 和
runner dry-run；不运行 1-epoch smoke。

- [ ] **步骤 2：启动正式矩阵**

使用 GPU 0/1/2、默认每卡 5 个并发任务运行 40 个 100-epoch jobs，runner 自动恢复完整
结果并记录失败任务。

- [ ] **步骤 3：审计结果**

确认 40/40 config、history、metrics 和单-rate prediction NPZ 完整；独立从 prediction
重算 W-F1，核对每个 test mask SHA，并检查单符号/非有限输出。

- [ ] **步骤 4：汇总比较**

按 rate 和 seed 汇总 fixed-rate，比较既有 mixed-rate Slot Missing-M3；明确 SDR-GNN 与
CaM-HG 的协议差异，不进行不配对显著性声明。

- [ ] **步骤 5：提交并推送**

只提交源码、测试、JSON/Markdown 和小型日志摘要，不提交 checkpoint 或 prediction NPZ；
按 Lore 协议提交并推送 `feature/m3-jepa-gcnet`。
