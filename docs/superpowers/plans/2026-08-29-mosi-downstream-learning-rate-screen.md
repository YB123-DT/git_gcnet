# MOSI 下游 Learning Rate 筛选实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 Remote Experiment Execution 执行此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不修改模型和 Loss 的情况下，筛选 Slot Missing-M3 + GCNet 的 MOSI 下游 Adam learning rate。

**架构：** 继承 `lr=1e-3` Control，只并行运行 seed66 的 `3e-4、5e-4、2e-3`。仅根据 validation 八-rate W-F1 均值选择候选，再决定是否扩 seeds 67--70。

**技术栈：** 现有 `gcnet_missing_m3.train_gcnet` CLI、biggpu、冻结 MOSI feature bank。

---

### 任务 1：锁定并启动三个 seed66 任务

**文件：**
- 读取：`experiments/missing_m3_mosi_all_rates_20260828/results/formal/seed_66/config.json`
- 输出：`experiments/missing_m3_mosi_lr_screen_20260829/results/screen/`

- [x] 核验继承 Control 配置为 Slot、Regression、all-rates、100 epochs、hidden 200、batch 32、`lr=1e-3`。
- [x] GPU 0/1/2 分别运行 `--lr 0.0003`、`--lr 0.0005`、`--lr 0.002`；其余参数逐项相同。
- [x] 每个任务必须产生 100 条 history、metrics 和 8 个 prediction NPZ。

### 任务 2：只按 validation 选择 LR

**文件：**
- 创建：`experiments/missing_m3_mosi_lr_screen_20260829/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_lr_screen_20260829/results/SCREEN_SUMMARY.json`

- [x] 从三个新 history 与继承 Control history 读取 `best_validation_mean_weighted_f1`。
- [x] 在读取 test 汇总前锁定 validation 最优 LR。
- [x] `5e-4` 以 79.482% validation 胜出；Control 为 77.887%。

### 任务 3：候选扩展或关闭

- [x] 筛选 PASS，运行 `5e-4` 的 seeds 67--70，Control 继续继承。
- [x] 根据用户明确要求，另补 `3e-4` seeds 67--70，标记为 test-observed 后追加的 exploratory control。
- [x] 独立重算 88 个 NPZ W-F1、核验 mask SHA，并记录逐 rate/seed delta。
- [x] 按 Lore protocol 提交并推送 `github feature/m3-jepa-gcnet`。
