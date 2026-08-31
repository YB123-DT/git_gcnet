# Differential-LR JEPA Transfer 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 检验 10 倍较小的预训练模块学习率能否避免 Stage 2 快速遗忘，同时让新 completion/readout 充分适配情绪任务。

**架构：** 从现有 `jepa-only` checkpoint 加载 Student、GCNet、Predictor 与 Teacher。Optimizer 将已加载的可训练模块置于 `5e-5` 参数组，将重新初始化的 `missing_latent_fusion` 与情绪 readout 置于 `5e-4` 参数组；保持 joint emotion+JEPA objective、EMA、mask 与其他配置不变。

**技术栈：** Python 3.8、PyTorch Adam、pytest、现有 MOSI cyclic trainer。

---

### 任务 1：参数分组 TDD

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] 编写失败测试：要求 `missing_predictor.*`、`graph_net_temporal.*` 属于 `5e-5` 组，`missing_latent_fusion.*`、`smax_fc.*` 属于 `5e-4` 组，Teacher 不进入 optimizer。
- [ ] 用远程官方 Python 运行 focused test，确认因缺少 `pretrained_learning_rate` 和参数分组 helper 而 FAIL。
- [ ] 在 `TrainConfig` 和 CLI 增加 `pretrained_learning_rate`；实现 `_optimizer_parameter_groups()`，只允许在 `joint + initial_backbone_checkpoint` 下启用。
- [ ] 将每组学习率、参数数量和参数名前缀写入 metrics provenance。
- [ ] 运行 focused tests，确认 PASS。

### 任务 2：完整回归验证与代码提交

**文件：**
- 测试：`tests/test_missing_m3.py`

- [ ] 运行完整 `tests/test_missing_m3.py`，要求全部 PASS。
- [ ] 运行 `git diff --check`，确认未修改 unrelated MOSEI 目录。
- [ ] 使用 Lore commit 提交代码与测试，记录只改变 optimizer learning-rate grouping。

### 任务 3：MOSI seed-66 正式判别

**文件：**
- 创建：`experiments/missing_m3_mosi_differential_lr_20260831/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_differential_lr_20260831/SUMMARY.json`
- 复制：config/history/metrics，不复制 checkpoint。

- [ ] 继承 Stage 1 seed-66 checkpoint，运行 joint completion：pretrained LR `5e-5`，fresh LR `5e-4`，JEPA weight `0.1`。
- [ ] 对齐 inherited cyclic control 79.013458，核算 8-rate mean 与逐 rate delta。
- [ ] 只有均值超过 control 且高 missing rates 不恶化时扩 seeds 67–70；否则停止。
- [ ] 归档、提交并推送 GitHub 功能分支。

