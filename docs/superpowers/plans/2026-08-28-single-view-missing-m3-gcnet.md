# Single-View Missing-M3 GCNet 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现一次 forward 的 Observed-Set Fusion + GCNet + 训练期 M3/EMA，并用一个 mixed-rate checkpoint 测试 IEMOCAPSix 八个 missing rates。

**架构：** 三个 observed modality projectors 经 mask-aware set mean 和 pattern embedding 构造 GCNet node；原 Temporal/Speaker graph backbone 输出分类 hidden；向量化六方向 M3 MMoE 仅在训练时预测真实缺失模态的 EMA latent。

**技术栈：** Python、PyTorch、PyTorch Geometric、pytest；正式实验使用 biggpu 的 GCNet official 环境。

---

## 文件结构

- 创建 `gcnet_missing_m3/__init__.py`：新版本包入口。
- 创建 `gcnet_missing_m3/model.py`：Observed-Set Encoder、M3 MMoE、EMA teacher 与 GCNet 集成。
- 创建 `gcnet_missing_m3/loss.py`：按真实缺失 target 聚合的 SmoothL1/InfoNCE。
- 创建 `gcnet_missing_m3/mixed_rate.py`：八 rate 的 train batch 调度与 validation/test 汇总。
- 创建 `gcnet_missing_m3/train_gcnet.py`：复用现有 loader/mask/metric 的单视图训练入口。
- 创建 `tests/test_missing_m3.py`：模型、泄漏、路由、EMA 与 mixed-rate 单测。
- 创建 `experiments/missing_m3_iemocap6_20260828/EXPERIMENT.md`：实验协议、命令、状态与结果。

### 任务 1：Observed-Set Encoder 与 GCNet 接入

**文件：** `tests/test_missing_m3.py`、`gcnet_missing_m3/model.py`

- [ ] 编写失败测试：七种非空 pattern 输出 `[L,B,d]`，padding 为零，修改缺失 feature block 不改变输出。
- [ ] 运行 `pytest -q tests/test_missing_m3.py -k observed_set`，确认因模块不存在而失败。
- [ ] 实现 `ObservedSetEncoder.forward(features, availability, umask)`：split raw blocks、模态 projector、availability 置零、set mean、pattern embedding、fusion MLP。
- [ ] 实现 `MissingM3GraphModel.encode_incomplete(...)`：用 fused node 调用 GCNet 原 recurrent/Temporal/Speaker path。
- [ ] 重跑同一测试并提交。

测试接口：

```python
node, latents = encoder(features, availability, umask)
assert node.shape == (length, batch, latent_dim)
assert torch.equal(node[~umask.T.bool()], torch.zeros_like(node[~umask.T.bool()]))
```

### 任务 2：向量化 M3 Predictor、EMA 与损失

**文件：** `tests/test_missing_m3.py`、`gcnet_missing_m3/model.py`、`gcnet_missing_m3/loss.py`

- [ ] 编写失败测试：A/T/V/AT/AV/TV target 选择正确，双 source 预测等于两方向均值，ATV JEPA loss 为零。
- [ ] 编写失败测试：teacher 参数无梯度，`update_teacher(0.9)` 精确满足 EMA 公式。
- [ ] 运行定向测试确认正确失败。
- [ ] 实现六方向批量 MMoE；每个方向只按 Boolean index 取一批样本，不写 time/batch 双循环。
- [ ] 实现 `missing_m3_loss`：SmoothL1 对真实 missing targets 计算；每个 target group 至少两个样本时加入对称 InfoNCE。
- [ ] 重跑测试并提交。

核心聚合：

```python
for target in range(3):
    for source in range(3):
        if source == target:
            continue
        selected = valid & availability[..., source].bool() & ~availability[..., target].bool()
        reg, cl = predictor(source_latent[selected], context[selected], source, target)
        reg_sum[target].index_add_(0, flat_index[selected], reg)
        cl_sum[target].index_add_(0, flat_index[selected], cl)
```

### 任务 3：Mixed-rate 生命周期

**文件：** `tests/test_missing_m3.py`、`gcnet_missing_m3/mixed_rate.py`、`gcnet_missing_m3/train_gcnet.py`

- [ ] 编写失败测试：连续八个 train batches 覆盖八个 rate 各一次；epoch 变化只平移起点；相同 seed/fold/epoch 可复现。
- [ ] 编写失败测试：checkpoint selector 使用八 rate validation W-F1 算术均值而非单 rate 或 test peak。
- [ ] 实现 `BalancedBatchRateSchedule` 与八套 `ConversationMaskSchedule`。
- [ ] 实现训练循环：每 batch 一个 rate、一次模型 forward；每 epoch 八 rate validation；保存最佳 state；最终八 rate test。
- [ ] 保存 config、history、per-rate metrics、predictions、mask hashes 与 checkpoint。
- [ ] 重跑测试并提交。

### 任务 4：一次性验证与正式运行

**文件：** `experiments/missing_m3_iemocap6_20260828/EXPERIMENT.md`

- [ ] 本地运行 `pytest -q tests/test_missing_m3.py`，要求全部通过。
- [ ] 在 biggpu 检查 GPU 与 official Python；同步分支代码。
- [ ] 只运行一个 seed、一个 batch 的 GPU forward/backward，要求 loss/gradient finite。
- [ ] 立即启动 seeds 66–70；每张可用 GPU 至多三个任务，不执行 Original。
- [ ] 拉回结果，核对每个 seed 只有一个 checkpoint 且包含八个 test-rate 指标。
- [ ] 汇总每 rate 五 seed weighted F1、macro F1、accuracy、均值和标准差，并更新实验 MD。
