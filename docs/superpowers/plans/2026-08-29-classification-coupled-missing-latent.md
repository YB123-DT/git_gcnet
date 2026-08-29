# Classification-Coupled Missing-Latent 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让训练期 Missing-M3 Predictor 在不完整推理时保留，并以预测 latent residual 修正 emotion hidden。

**架构：** 同一次 GCNet hidden 同时供 Predictor 和 classifier 使用；target-specific 零初始化投影把实际缺失目标的 regression prediction 平均成 residual。Teacher 和 JEPA loss 仍只在训练存在。

**技术栈：** Python、PyTorch、pytest、现有 Missing-M3 远程执行入口。

---

### 任务 1：TDD 锁定 completion 合同

**文件：**
- 修改：`tests/test_missing_m3.py`

- [ ] 写失败测试：不完整 inference 即使 `predict_missing=False` 也调用 Predictor，但返回的 prediction artifact 仍为 `None`。
- [ ] 写失败测试：ATV 与 padding residual 精确为零；两个缺失 target 的 residual 取平均。
- [ ] 写失败测试：默认模型 state-dict 不增加 completion keys；CLI 显式开启 treatment。
- [ ] 通过统一 wrapper 运行 focused test，确认因 completion API 缺失而正确失败。

### 任务 2：最小实现

**文件：**
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 测试：`tests/test_missing_m3.py`

- [ ] 实现 `MissingLatentResidualFusion`：三个 target-specific LN+Linear、线性层零初始化、tanh、target average、padding zero。
- [ ] `MissingM3GraphModel` 仅在开关开启时实例化新参数；训练复用已有 predictions，推理内部执行 Predictor但不返回 artifact。
- [ ] 将 appended config 与 `--classification-completion` 贯通，不改变旧 positional config。
- [ ] 运行 Missing-M3 与相关完整测试。

### 任务 3：远程验证和五种子实验

**文件：**
- 创建：`experiments/missing_m3_mosi_classification_completion_20260829/EXPERIMENT.md`
- 生成：`experiments/missing_m3_mosi_classification_completion_20260829/results/`

- [ ] 用 `scripts/remote_missing_m3.sh` preflight、同步、测试；official 环境仅执行一次 1-epoch 入口验证。
- [ ] GPU 0/1/2/3/5 各运行 seeds 66--70；不重跑 control。
- [ ] 回传 40 NPZ 和 provenance，重算指标并核验 40/40 mask hash。
- [ ] 汇总八-rate/high-missing 配对差，提交并推送用户 GitHub remote。

