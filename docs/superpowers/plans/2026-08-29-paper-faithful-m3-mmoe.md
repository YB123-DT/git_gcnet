# Paper-Faithful M3 MMoE 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Missing-M3 增加论文双 gate 与官方 task-embedding/norm/residual 组合的可切换 MMoE，并完成 MOSI 五种子配对实验。

**架构：** 默认 `dual-gate` 完全保持旧 checkpoint；`paper-faithful` 让 reg/cl 使用独立 task embedding、独立 gate 与独立 norm/residual，同时共享 expert 参数。路由统计通过非持久 buffer 暴露，不改变优化目标。

**技术栈：** Python、PyTorch、pytest、GCNet official remote environment。

---

### 任务 1：用失败测试锁定 MMoE 合同

**文件：**
- 修改：`tests/test_missing_m3.py`

- [ ] 编写测试，要求默认 `DualGateTopKMMoE` state-dict 不出现新 key，`paper-faithful` 出现两个 task embedding 和两个 norm。
- [ ] 编写确定性测试：将 experts 置零、heads 置 identity，验证输出等于 branch input，从而证明 residual 存在。
- [ ] 编写路由测试：固定 gate logits，验证权重为 full-softmax 后 Top-K、不二次归一化，并验证 selection/probability/entropy 统计。
- [ ] 运行 `python -m pytest -q tests/test_missing_m3.py -k 'paper_faithful or routing_statistics'`，预期因 `mmoe_variant` 尚不存在而失败。

### 任务 2：实现最小 MMoE 变体

**文件：**
- 修改：`gcnet_missing_m3/model.py:301-395`
- 测试：`tests/test_missing_m3.py`

- [ ] 为 `DualGateTopKMMoE` 增加 `variant`，只在 `paper-faithful` 下实例化 task embedding 与 branch norm。
- [ ] 保留两个 gate 和共享 experts；实现 full-softmax Top-K、branch-specific expert forward、LayerNorm、GELU 与 residual。
- [ ] 增加非持久 routing buffers，以及 `reset_routing_statistics()`、`routing_statistics()`。
- [ ] 运行 focused tests，预期全部通过；再运行 `python -m pytest -q tests/test_missing_m3.py`，修复所有回归。

### 任务 3：贯通配置与 CLI

**文件：**
- 修改：`gcnet_missing_m3/model.py:368-542`
- 修改：`gcnet_missing_m3/train_gcnet.py:40-790`
- 修改：`tests/test_missing_m3.py`

- [ ] 先写失败测试：CLI 接受 `--mmoe-variant paper-faithful`，`TrainConfig` 将其传入 `MissingM3GraphModel`。
- [ ] 增加 `mmoe_variant: str = "dual-gate"`；保持字段追加顺序，避免破坏旧 positional config。
- [ ] 将配置贯通到 `ContextualM3Predictor` 与 `DualGateTopKMMoE`。
- [ ] 运行 focused test 和 Missing-M3 全测试。

### 任务 4：远程验证与正式实验

**文件：**
- 创建：`experiments/missing_m3_mosi_paper_mmoe_20260829/EXPERIMENT.md`
- 生成：`experiments/missing_m3_mosi_paper_mmoe_20260829/results/`

- [ ] 同步修改文件到 biggpu 正确的 Single-View 工作目录。
- [ ] 在 official Python 环境运行 focused tests 和一个真实 MOSI forward/backward，确认 loss、梯度与路由统计有限。
- [ ] 使用 GPU 0--2 启动 seeds 66--70；每张可用卡并发不超过既有稳定配置，训练一个 checkpoint 并测试八个 rate。
- [ ] 拉回 checkpoint provenance、history、八-rate NPZ 和 summary；继承当前 `dual-gate` 五种子结果，计算逐 rate、八-rate、高缺失均值和逐 seed delta。
- [ ] 以 Lore commit 提交代码、测试、规格与实验记录，并推送当前 GitHub 分支。

