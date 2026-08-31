# Frozen JEPA Completion 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加一个可复现的 frozen-completion Stage 2，只训练缺失 latent 投影与情绪 readout，并完成 MOSI seed-66 判别实验。

**架构：** 从现有 `jepa-only` checkpoint 加载 online encoder、GCNet、Predictor 与 Teacher；冻结全部已加载表示参数，重新初始化并训练 `missing_latent_fusion` 和情绪 readout。Stage 2 仅使用情绪损失，Predictor 参与前向但 Teacher/EMA 不参与，并以冻结参数训练前后 SHA256 证明表示没有漂移。

**技术栈：** Python 3.8、PyTorch、pytest、现有 `gcnet_missing_m3` trainer、MOSI cyclic missing-rate protocol。

---

### 任务 1：锁定冻结边界和生命周期

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

增加测试，要求 `_configure_frozen_completion_probe(model)` 返回的可训练参数仅允许以下前缀：

```python
allowed = (
    "smax_fc.",
    "conditioned_readout.",
    "affine_readout.",
    "missing_latent_fusion.",
)
assert trainable
assert all(name.startswith(allowed) for name in trainable)
assert "missing_predictor.context_projection.weight" in frozen
assert "graph_net_temporal.conv1.weight" in frozen
```

再增加配置约束测试：`frozen-completion` 必须同时提供 `initial_backbone_checkpoint` 并启用 `classification_completion`。

- [ ] **步骤 2：运行测试验证失败**

运行远程官方 Python 的 focused pytest。预期：因缺少 objective/helper 而 FAIL。

- [ ] **步骤 3：实现最少生命周期代码**

在 `TrainConfig` 支持的 objective 中加入 `frozen-completion`。实现：

```python
def _configure_frozen_completion_probe(model):
    allowed = _FROZEN_COMPLETION_TRAINABLE_PREFIXES
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(allowed))
    return {
        "trainable_parameter_names": [
            name for name, value in model.named_parameters()
            if value.requires_grad
        ],
        "frozen_parameter_names": [
            name for name, value in model.named_parameters()
            if not value.requires_grad
        ],
    }
```

`run_experiment()` 对该 objective 使用 `include_jepa_modules=True` 加载 Stage 1，但训练循环设置 `train_emotion=True`、`train_jepa=False`。

- [ ] **步骤 4：运行 focused tests 验证通过**

预期：冻结边界、配置约束和生命周期测试全部 PASS。

### 任务 2：增加冻结不变量证明

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

对冻结参数计算 deterministic SHA256；只更新允许的 trainable 参数后，要求 hash 不变，修改任一冻结 tensor 后要求 hash 改变。

- [ ] **步骤 2：实现参数哈希**

实现按参数名排序、同时写入 name/dtype/shape/raw bytes 的：

```python
def _parameter_subset_sha256(model, names):
    digest = hashlib.sha256()
    parameters = dict(model.named_parameters())
    for name in sorted(names):
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
```

在训练前后记录 `frozen_parameter_sha256_before/after`；若不一致立即报错。将 trainable/frozen 参数数量和 hash 写入 metrics。

- [ ] **步骤 3：运行 focused tests 验证通过**

预期：hash 不变量测试 PASS。

### 任务 3：完整回归验证

**文件：**
- 测试：`tests/test_missing_m3.py`

- [ ] **步骤 1：运行完整测试**

使用 biggpu 官方 Python，并补入已有 pytest site-packages，运行：

```bash
pytest -q tests/test_missing_m3.py
```

预期：全部 PASS，无既有 joint、jepa-only、emotion-only 回归。

- [ ] **步骤 2：检查 diff**

运行 `git diff --check`，确认无格式错误；确认未修改 unrelated MOSEI 结果目录。

### 任务 4：运行 MOSI seed-66 判别实验

**文件：**
- 创建：`experiments/missing_m3_mosi_frozen_completion_20260831/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_frozen_completion_20260831/SUMMARY.json`
- 复制：该实验的 config/history/metrics（不复制 checkpoint）

- [ ] **步骤 1：同步代码并启动正式任务**

复用 Stage 1：

```text
/data2/yb/remote_experiments/missing_m3_mosi_two_stage_20260831/stage1/seed_66/best.pt
```

使用 `--training-objective frozen-completion --classification-completion`，其余参数与 cyclic seed-66 control 一致，在可用且非 GPU4 的卡运行。

- [ ] **步骤 2：核验结果**

检查：最佳 epoch、8-rate W-F1、均值、逐 rate delta、checkpoint provenance，以及 frozen SHA before/after 完全一致。

- [ ] **步骤 3：执行门槛决策**

只在均值明确优于 79.013458 且高 missing rates 不恶化时扩展 seeds 67–70；否则以 seed 66 关闭该路线。

### 任务 5：归档和交付

**文件：**
- 修改：实验文档与汇总 JSON

- [ ] **步骤 1：提交代码与测试**

使用 Lore commit 记录冻结边界、拒绝的 Predictor-only freeze 及完整测试证据。

- [ ] **步骤 2：提交实验结果**

记录 control 来源、Stage 1 SHA、所有 rate 分数与停止/扩展决定。

- [ ] **步骤 3：推送 GitHub 并核验**

推送 `feature/missing-m3-sdr-backbone` 到 `github` remote，用 `git ls-remote` 核验远端 commit。
