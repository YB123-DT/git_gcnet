# CMU-MOSI 软有序分类头实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 Missing-M3/GCNet 主干、mask、JEPA 和回归默认行为的前提下，为 CMU-MOSI 增加单 signed-logit 的 soft-ordinal 分类模式，并完成五种子八 missing-rate 配对实验。

**架构：** `mosi_task_mode=soft-ordinal` 继续构造单输出 `smax_fc`，将连续标签线性映射到 `[0,1]` 后使用 `BCEWithLogits`；指标阶段排除零标签并固定以 logit 零为正负阈值。现有 `gcnet_missing_m3` 保留为 regression 版本，新建 `gcnet_missing_m3_soft_ordinal` 独立入口；两者共享 model/data/mask/JEPA/GCNet 和训练生命周期，不复制 backbone。旧 `regression` 与 `binary` 分支逐字保留，treatment 与 regression 参数量相同。

**技术栈：** Python、PyTorch、NumPy、scikit-learn、pytest、PyTorch Geometric、biggpu V100。

---

## 文件结构

- 修改 `gcnet_missing_m3/train_gcnet.py`：task contract、soft target helper、loss、prediction collection、metrics 与 NPZ provenance。
- 创建 `gcnet_missing_m3_soft_ordinal/__init__.py`：声明独立版本包。
- 创建 `gcnet_missing_m3_soft_ordinal/train_gcnet.py`：锁定 soft-ordinal 的薄入口，复用共享训练器。
- 创建 `gcnet_missing_m3_soft_ordinal/tests/test_train_gcnet.py`：锁定版本入口不能退回 regression/binary。
- 修改 `tests/test_missing_m3.py`：contract、soft target、zero/padding、metric threshold、artifact 和旧模式回归测试。
- 创建 `experiments/missing_m3_mosi_soft_ordinal_head_20260831/EXPERIMENT.md`：锁定协议、继承 control、五种子结果与结论。
- 创建 `experiments/missing_m3_mosi_soft_ordinal_head_20260831/results/`：仅保存轻量 config/history/metrics/prediction NPZ 与汇总，不保存 checkpoint。

### 任务 1：锁定 task contract 与单输出形状

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py:120-132`
- 修改：`gcnet_missing_m3/train_gcnet.py:875-885`
- 创建：`gcnet_missing_m3_soft_ordinal/__init__.py`
- 创建：`gcnet_missing_m3_soft_ordinal/train_gcnet.py`
- 创建：`gcnet_missing_m3_soft_ordinal/tests/test_train_gcnet.py`

- [ ] **步骤 1：编写失败的 contract 与 CLI 测试**

在现有 MOSI task mode 测试旁加入：

```python
def test_mosi_soft_ordinal_contract_uses_single_signed_logit():
    contract = train_gcnet._resolve_task_contract(
        "CMUMOSI", "soft-ordinal"
    )
    assert contract["task"] == "soft-ordinal"
    assert contract["num_classes"] == 1


def test_mosi_soft_ordinal_cli_is_explicit_and_mosi_only():
    required = [
        "--audio-feature", "a", "--text-feature", "t",
        "--video-feature", "v", "--output-dir", "out",
    ]
    args = build_parser().parse_args(
        required + ["--dataset", "CMUMOSI", "--mosi-task-mode", "soft-ordinal"]
    )
    assert args.mosi_task_mode == "soft-ordinal"
    with pytest.raises(ValueError, match="CMUMOSI"):
        train_gcnet._resolve_task_contract("IEMOCAPSix", "soft-ordinal")


def test_soft_ordinal_version_entry_injects_locked_task_mode(monkeypatch):
    captured = {}

    def shared_main(argv=None):
        captured["argv"] = list(argv)

    monkeypatch.setattr(soft_train.base_train, "main", shared_main)
    soft_train.main(["--audio-feature", "a"])
    assert captured["argv"][-2:] == [
        "--mosi-task-mode", "soft-ordinal"
    ]
```

- [ ] **步骤 2：运行测试并确认正确红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py \
  -k 'soft_ordinal_contract or soft_ordinal_cli'
```

预期：FAIL，`soft-ordinal` 被现有 mode validation 或 argparse choices 拒绝；不是环境收集失败。

- [ ] **步骤 3：实现最小 task contract**

将共享 mode validation 和 CLI choices 扩展为：

```python
if mode not in ("regression", "binary", "soft-ordinal"):
    raise ValueError("unsupported MOSI task mode: {}".format(mode))

if mode in ("binary", "soft-ordinal"):
    if dataset != "CMUMOSI":
        raise ValueError("{} task mode is only supported for CMUMOSI".format(mode))
if mode == "binary":
    contract.update(task="binary", num_classes=2)
elif mode == "soft-ordinal":
    contract.update(task="soft-ordinal", num_classes=1)
```

CLI：

```python
choices=("regression", "binary", "soft-ordinal")
```

让共享 `main` 可接收显式 argv，但无参数调用保持原语义：

```python
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
```

新版本入口只注入锁定 task mode：

```python
def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--mosi-task-mode" in arguments:
        raise ValueError("soft-ordinal version owns mosi_task_mode")
    base_train.main([
        *arguments, "--mosi-task-mode", "soft-ordinal"
    ])
```

不要复制 `MissingM3GraphModel`、train loop、dataset 或 mask 代码。

- [ ] **步骤 4：运行 focused test 确认通过**

重复步骤 2，预期 2 个测试 PASS。

### 任务 2：用连续标签构造 soft BCE

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py:303-347`

- [ ] **步骤 1：编写失败的软目标与 loss 测试**

```python
def test_mosi_soft_ordinal_targets_preserve_order_and_clip_range():
    labels = torch.tensor([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0])
    expected = torch.tensor([0.0, 0.0, 1 / 3, 0.5, 2 / 3, 1.0, 1.0])
    ASSERT_CLOSE(train_gcnet._mosi_soft_targets(labels), expected)


def test_mosi_soft_ordinal_loss_includes_zero_and_excludes_padding():
    logits = torch.tensor(
        [[[0.0]], [[1.0]], [[-2.0]], [[99.0]]], requires_grad=True
    )
    labels = torch.tensor([[-3.0, 0.0, 3.0, -3.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 1.0, -2.0]),
        torch.tensor([0.0, 0.5, 1.0]),
    )
    actual = _task_loss(
        "CMUMOSI", logits, labels, umask,
        mosi_task_mode="soft-ordinal",
    )
    ASSERT_CLOSE(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[3].item() == 0.0
```

- [ ] **步骤 2：运行测试并确认红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py \
  -k 'soft_ordinal_targets or soft_ordinal_loss'
```

预期：FAIL，缺少 `_mosi_soft_targets` 或 `_task_loss` 尚未识别该 task。

- [ ] **步骤 3：实现纯 helper 与最小 loss 分支**

在 `_task_loss` 前新增：

```python
def _mosi_soft_targets(labels: torch.Tensor) -> torch.Tensor:
    return (labels.clamp(min=-3.0, max=3.0) + 3.0) / 6.0
```

在 classification/binary 分支之前加入：

```python
if task == "soft-ordinal":
    prediction = logits.transpose(0, 1).reshape(-1)
    target = _mosi_soft_targets(
        labels.reshape(-1).to(dtype=prediction.dtype)
    )
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        prediction[selected], target[selected]
    )
```

不要修改 regression MSE/SmoothL1 或 binary CE 代码。

- [ ] **步骤 4：运行 focused tests 确认通过**

重复步骤 2；预期 PASS。

### 任务 3：固定零阈值并保存 signed-logit provenance

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py:360-423`
- 修改：`gcnet_missing_m3/train_gcnet.py:580-665`

- [ ] **步骤 1：编写失败的 prediction/metric 测试**

```python
def test_mosi_soft_ordinal_collection_uses_zero_logit_threshold():
    logits = torch.tensor([[[-0.2]], [[0.0]], [[0.4]], [[9.0]]])
    labels = torch.tensor([[-1.0, 0.0, 2.0, -2.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    predictions, metric_labels, continuous = train_gcnet._collect_predictions(
        "CMUMOSI", logits, labels, umask,
        mosi_task_mode="soft-ordinal",
    )
    assert predictions.tolist() == [0, 1]
    assert metric_labels.tolist() == [0, 1]
    assert continuous.tolist() == [-1.0, 2.0]


def test_mosi_soft_ordinal_metrics_are_binary_without_regression_fields():
    result = _metrics(
        "CMUMOSI",
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        mosi_task_mode="soft-ordinal",
    )
    assert set(result) == {"accuracy", "weighted_f1", "macro_f1"}
    assert result["accuracy"] == pytest.approx(0.75)
```

- [ ] **步骤 2：运行测试并确认红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py \
  -k 'soft_ordinal_collection or soft_ordinal_metrics'
```

预期：FAIL，当前 collection 把单 logit 当 regression 数值，metrics 也未走二分类数组路径。

- [ ] **步骤 3：实现固定阈值 collection 与 metrics**

在 `_metrics` 中将 direct class-array 分支改为：

```python
if task in ("classification", "binary", "soft-ordinal"):
    return {
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }
```

在 `_collect_predictions` 中保持 ERC/binary 不变，并加入：

```python
if task == "soft-ordinal":
    signed_logits = logits.squeeze(-1).transpose(0, 1)
    selected = umask.bool() & labels.ne(0)
    predicted = signed_logits.gt(0).long()
    metric_labels = labels.gt(0).long()
```

`evaluate_rate` 的 artifact 在 soft-ordinal 模式额外保存有效 nonzero utterance 的
`signed_logits`，同时保持 `predictions`/`labels` 为二值数组、`continuous_labels` 为原
连续标签。不要加入可调 threshold 参数。

- [ ] **步骤 4：增加 artifact 一致性测试并运行**

扩展 mock evaluate 测试，要求：

```python
assert set(npz.files) >= {
    "predictions", "labels", "continuous_labels",
    "signed_logits", "availability",
}
assert np.array_equal(npz["predictions"], npz["signed_logits"] > 0)
```

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py -k 'soft_ordinal'
```

预期：全部 soft-ordinal tests PASS。

### 任务 4：证明旧任务路径无回归并提交实现

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：增加参数与旧模式等价测试**

测试构造三个 task contract 对应的模型，要求：

```python
assert regression_model.smax_fc.out_features == 1
assert soft_ordinal_model.smax_fc.out_features == 1
assert binary_model.smax_fc.out_features == 2
assert count_parameters(regression_model) == count_parameters(soft_ordinal_model)
```

复用现有 `test_default_and_explicit_regression_helpers_are_equivalent`，并确认旧 binary
loss/collection/metrics tests 不修改断言。

- [ ] **步骤 2：运行完整相关测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py tests/test_mosi_conditioned_readout.py
/data2/yb/reproduction_envs/s0/bin/python -m py_compile \
  gcnet_missing_m3/train_gcnet.py tests/test_missing_m3.py
git diff --check
```

预期：退出码均为 0；不存在 regression/binary test 失败。

- [ ] **步骤 3：提交代码与测试**

```bash
git add gcnet_missing_m3/train_gcnet.py tests/test_missing_m3.py \
  gcnet_missing_m3_soft_ordinal \
  docs/superpowers/plans/2026-08-31-mosi-soft-ordinal-classification-head.md
git commit -m "让 MOSI 分类边界保留连续情感顺序"
```

提交必须使用 Lore trailers，明确旧 hard CE 已失败、固定零阈值以及尚未完成正式训练。

### 任务 5：远程一次验证后启动五种子正式实验

**文件：**
- 创建：`experiments/missing_m3_mosi_soft_ordinal_head_20260831/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_soft_ordinal_head_20260831/results/`

- [ ] **步骤 1：只同步修改文件并运行一次 focused remote test**

```bash
scripts/remote_missing_m3.sh sync \
  gcnet_missing_m3/train_gcnet.py tests/test_missing_m3.py \
  gcnet_missing_m3_soft_ordinal
scripts/remote_missing_m3.sh test \
  tests/test_missing_m3.py gcnet_missing_m3_soft_ordinal/tests -q \
  -k 'soft_ordinal'
```

不重复 `preflight`；远程 Python 路径和依赖已经由仓库既有记录验证。

- [ ] **步骤 2：在真实 MOSI batch 上做一次 forward/backward 集成**

使用训练入口运行 `1 epoch`、seed 66、`soft-ordinal` 到临时目录，只验证真实 dataloader、
loss、NPZ 与 backward；成功后删除临时 checkpoint，不再运行其他 smoke。

- [ ] **步骤 3：并行启动 seeds 66--70**

使用健康 GPU `2,3,5,6,7`，每张卡一个 seed；若实际只有三张卡空闲，则首波
`66,67,68`，完成后自动续跑 `69,70`。每个命令固定：

```text
dataset=CMUMOSI
fold=1
epochs=100
batch_size=32
train_rate_mode=all
mosi_task_mode=soft-ordinal
fusion_type=slot
representation_type=slot
graph_branch_mode=both
hidden/window/lr/l2/jepa_weight = paired regression anchor 的逐字段值
training_module=gcnet_missing_m3_soft_ordinal.train_gcnet
```

不得重跑 Original、regression 或旧 binary。输出放入远程独立目录
`/data2/yb/remote_experiments/missing_m3_mosi_soft_ordinal_head_20260831/`，每个 seed
保存 `status.json` 与 `train.log`，完成判定只依赖 `metrics.json` 和 8 个 prediction NPZ。

- [ ] **步骤 4：审计并汇总结果**

对每个 seed 检查：history 100 epochs、validation-only best epoch、8/8 test NPZ、两类
预测、signed-logit finite/std、mask SHA 与 paired regression 一致。生成逐 seed/rate、
mean/std、paired delta 和 gate verdict；任何失败任务只补失败 seed，不重跑成功任务。

- [ ] **步骤 5：同步轻量结果、Lore commit 与推送**

只同步 `config.json`、`history.json`、`metrics.json`、prediction NPZ、汇总 JSON 与实验
Markdown；不上传 checkpoint 或训练日志。提交中记录实际测试、五种子完成度和所有已知
风险，随后推送当前 GitHub branch。

## 自检

- 规格中的单输出、软标签、zero 语义、固定阈值、参数匹配、旧模式兼容、五种子和 NPZ
  审计均有对应任务。
- 计划不包含 margin、focal、threshold search、class weighting、dual head 或 backbone
  修改。
- `_mosi_soft_targets`、`soft-ordinal` task 名称与 artifact key `signed_logits` 在所有任务中
  一致。
- regression 和 treatment 具有两个独立启动入口，但不复制 shared backbone/train loop。
- 只在第一次真实集成使用 1 epoch smoke；不会重复环境检查、Original 或 paired control。
