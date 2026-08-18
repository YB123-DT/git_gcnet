# GCNet Modality-JEPA 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 固定 seed=66，在完全隔离的目录中完成原始 GCNet 与 centered fixed-latent Modality-JEPA 的 IEMOCAP-Six missing-rate 0.0–0.7 五折实验。

**架构：** 原始 `gcnet/` 只接受 NumPy 兼容修复；新建 `gcnet_modality_jepa/`，保留 GCNet 分类与重建路径，从 500D conversation hidden 通过三个独立 MLP 预测被遮蔽的 A/T/V 训练折中心化 latent。实验脚本为每个方法和 missing rate 分配独立输出目录，并从 NPZ 重新汇总分类与预测诊断指标。

**技术栈：** Python 3.10、PyTorch 2.2、NumPy、scikit-learn、torch-geometric、pytest、Bash。

---

## 文件结构

- 修改 `gcnet/train_gcnet.py`：仅修复 `np.int`，不改变基线行为。
- 创建 `gcnet_modality_jepa/model.py`：复用 GCNet 图模型并添加三个 predictor。
- 创建 `gcnet_modality_jepa/loss.py`：mask-aware centered cosine loss。
- 创建 `gcnet_modality_jepa/targets.py`：训练 fold 的 A/T/V 均值统计。
- 创建 `gcnet_modality_jepa/metrics.py`：Real/Shuffle、标准差和 effective rank。
- 创建 `gcnet_modality_jepa/train_gcnet.py`：独立训练入口与诊断结果保存。
- 创建 `scripts/run_missing_sweep.py`：隔离运行和状态记录。
- 创建 `scripts/summarize_missing_sweep.py`：从保存结果计算五折指标。
- 创建 `tests/test_random_mask.py`、`tests/test_modality_jepa.py`、`tests/test_experiment_isolation.py`：回归、梯度和隔离测试。
- 创建两个 `experiments/*/EXPERIMENT.md`：持续记录命令、状态和结果。

### 任务 1：锁定原始 mask 行为并修复 NumPy 兼容

**文件：**
- 修改：`gcnet/train_gcnet.py:155-198`
- 创建：`tests/test_random_mask.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_random_mask_supports_numpy_124_and_keeps_one_view():
    np.random.seed(66)
    mask = random_mask(3, 128, 0.7)
    assert mask.shape == (128, 3)
    assert np.issubdtype(mask.dtype, np.integer)
    assert np.all(mask.sum(axis=1) >= 1)
```

- [ ] **步骤 2：验证红灯**

运行：`pytest -q tests/test_random_mask.py`
预期：FAIL，提示 NumPy 不再提供 `np.int`。

- [ ] **步骤 3：最小修复**

把三处 `.astype(np.int)` 改为 `.astype(np.int64)`，其余算法不动。

- [ ] **步骤 4：验证绿灯和边界率**

运行：`pytest -q tests/test_random_mask.py`
预期：0.0、0.1、0.7 均通过，且每行至少一个 1。

### 任务 2：实现 fold-only 中心统计

**文件：**
- 创建：`gcnet_modality_jepa/__init__.py`
- 创建：`gcnet_modality_jepa/targets.py`
- 创建：`tests/test_modality_jepa.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_fold_means_ignore_padding_and_excluded_session():
    means = compute_modality_means(loader, device="cpu")
    torch.testing.assert_close(means.audio, expected_audio)
    torch.testing.assert_close(means.text, expected_text)
    torch.testing.assert_close(means.visual, expected_visual)
```

- [ ] **步骤 2：验证红灯**

运行：`pytest -q tests/test_modality_jepa.py -k fold_means`
预期：FAIL，模块尚不存在。

- [ ] **步骤 3：实现统计接口**

实现 `ModalityMeans(audio, text, visual)` 和 `compute_modality_means(dataloader, device)`；使用 `qmask` 选择实际说话者特征、`umask` 排除 padding，在 CPU float64 累加后输出 float32。

- [ ] **步骤 4：验证绿灯**

运行：`pytest -q tests/test_modality_jepa.py -k fold_means`
预期：PASS。

### 任务 3：实现三个 predictor 和 mask-aware loss

**文件：**
- 创建：`gcnet_modality_jepa/model.py`
- 创建：`gcnet_modality_jepa/loss.py`
- 修改：`tests/test_modality_jepa.py`

- [ ] **步骤 1：编写输出、mask、空目标和梯度测试**

```python
assert preds.audio.shape == (seq_len, batch, 512)
assert preds.text.shape == (seq_len, batch, 1024)
assert preds.visual.shape == (seq_len, batch, 1024)
assert torch.isfinite(loss_without_missing)
assert loss_without_missing.item() == 0.0
assert target.grad is None
assert hidden.grad is not None
```

- [ ] **步骤 2：验证红灯**

运行：`pytest -q tests/test_modality_jepa.py -k 'predictor or loss or gradient'`
预期：FAIL，类和函数尚不存在。

- [ ] **步骤 3：实现最小模型和损失**

实现 `ModalityPredictions`、`ModalityPredictor`、继承/复用 `GraphModel` 的 `ModalityJEPAGraphModel`，以及：

```python
loss, per_modality = masked_centered_cosine_loss(
    predictions, full_features, availability_mask, umask, means
)
```

loss 只在 `availability_mask == 0` 且 `umask == 1` 位置计算；target 显式 `detach()`。

- [ ] **步骤 4：验证绿灯**

运行：`pytest -q tests/test_modality_jepa.py`
预期：全部通过。

### 任务 4：实现预测诊断

**文件：**
- 创建：`gcnet_modality_jepa/metrics.py`
- 修改：`tests/test_modality_jepa.py`

- [ ] **步骤 1：编写确定性指标测试**

构造正交 target 和完全正确 prediction，验证 Real cosine 为 1、固定 permutation 的 Shuffle 更低、effective rank 有限且常量 prediction 的 rank 为 1。

- [ ] **步骤 2：验证红灯**

运行：`pytest -q tests/test_modality_jepa.py -k metrics`
预期：FAIL，指标模块尚不存在。

- [ ] **步骤 3：实现指标**

实现固定 seed permutation、cosine、mean per-dimension std 和基于奇异值熵的 effective rank。少于两个样本时返回 `null` 诊断而不是 NaN。

- [ ] **步骤 4：验证绿灯**

运行：`pytest -q tests/test_modality_jepa.py -k metrics`
预期：PASS。

### 任务 5：建立独立训练入口

**文件：**
- 创建：`gcnet_modality_jepa/train_gcnet.py`
- 创建：`tests/test_experiment_isolation.py`

- [ ] **步骤 1：编写 CLI 与隔离测试**

测试 `--fold`、`--epochs`、`--jepa-weight`、`--output-dir`，确认原始与 JEPA 输出路径不能相同，并确认 `miss=0` 的 JEPA loss 为零。

- [ ] **步骤 2：验证红灯**

运行：`pytest -q tests/test_experiment_isolation.py`
预期：FAIL，新入口尚不存在。

- [ ] **步骤 3：实现训练入口**

从原入口保留数据、mask、GCNet loss 和评估语义；每 fold 在训练 loader 上计算一次 means；总损失为 `L_cls + L_rec + 0.1*L_jepa`；保存逐 epoch 分类结果、最佳 epoch 的 predictors/targets/masks 和 `fold_metrics.json`。

- [ ] **步骤 4：验证完整测试集**

运行：`pytest -q tests`
预期：全部通过。

### 任务 6：单折 smoke run

**文件：**
- 创建：`experiments/original_missing_sweep_seed66_20260818/EXPERIMENT.md`
- 创建：`experiments/modality_jepa_seed66_20260818/EXPERIMENT.md`

- [ ] **步骤 1：运行基线 miss=0.1、fold=1、2 epochs smoke**

运行原始入口的短运行包装命令，输出到 baseline smoke 子目录。
预期：训练、评估和 NPZ 保存完成。

- [ ] **步骤 2：运行 JEPA miss=0.1、fold=1、2 epochs smoke**

运行新入口，输出到 JEPA smoke 子目录。
预期：分类、重建、JEPA loss 均有限，predictor 有梯度，JSON 保存完成。

- [ ] **步骤 3：记录 smoke 证据**

在两个 `EXPERIMENT.md` 中记录命令、退出码、耗时和结果路径。

### 任务 7：实现并启动完整 sweep

**文件：**
- 创建：`scripts/run_missing_sweep.py`
- 创建：`scripts/summarize_missing_sweep.py`
- 修改：两个 `EXPERIMENT.md`

- [ ] **步骤 1：实现调度器 dry-run**

生成 16 个实验命令（2 methods × 8 rates），每个命令独立目录；检测 `nvidia-smi` 当前空闲卡，只使用无计算进程的 GPU。

- [ ] **步骤 2：验证 dry-run**

运行：`python scripts/run_missing_sweep.py --seed 66 --dry-run`
预期：16 个唯一输出目录，无路径冲突。

- [ ] **步骤 3：启动完整训练**

空闲 GPU 全部可用；每张卡同时只运行一个训练进程。先并行 baseline 和 JEPA 队列，再按空闲卡补充剩余 rate。

- [ ] **步骤 4：监控与恢复**

记录 PID、GPU、开始/结束时间和退出码。失败任务只重跑自身目录，不覆盖成功结果。

- [ ] **步骤 5：汇总并核验**

从保存预测重算 Weighted-F1、Macro-F1、Accuracy 和 JEPA diagnostics，生成逐 rate 对照表与 JSON；核对每组均为 5 folds。

### 任务 8：最终验证与实验记录

**文件：**
- 修改：两个 `EXPERIMENT.md`

- [ ] **步骤 1：运行测试和静态编译**

运行：`pytest -q tests && python -m compileall -q gcnet gcnet_modality_jepa scripts`
预期：退出码 0。

- [ ] **步骤 2：核验实验完整性**

检查 16 个 rate/method 目录、80 个 fold 结果、固定 seed=66、互不覆盖、无 NaN。

- [ ] **步骤 3：写入结论边界**

记录哪些 missing rate 有正/负增益、Real–Shuffle 是否为正、是否出现低秩预测，并明确单 seed 与官方 validation/test 重用的局限。

