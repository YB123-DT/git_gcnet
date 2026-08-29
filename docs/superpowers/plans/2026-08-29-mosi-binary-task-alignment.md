# CMU-MOSI 二分类任务对齐实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Missing-M3 增加向后兼容的 MOSI 无零二分类模式，以纯 Cross-Entropy 对照官方 regression-MSE，并完成 paired seed66 八 missing-rate 判别实验。

**架构：** `TrainConfig` 通过 `mosi_task_mode` 决定 MOSI classifier 输出维度和任务 helper。Binary 模式只在监督与指标阶段排除连续标签零值；完整 conversation、mask、Student/Teacher、GCNet 和 JEPA 数据流保持不变。Regression 为默认且不改变已有 state keys、输出或 RNG。

**技术栈：** Python、PyTorch、NumPy、scikit-learn、pytest、PyTorch Geometric、biggpu V100。

---

## 文件结构

- 修改 `gcnet_missing_m3/train_gcnet.py`：任务模式配置、CLI、模型输出维度、loss、prediction collection、artifact metadata。
- 修改 `tests/test_missing_m3.py`：binary label/loss/metrics、zero-context、回归兼容和 CLI 回归测试。
- 创建 `experiments/missing_m3_mosi_binary_task_20260829/EXPERIMENT.md`：协议、验证、seed66结果与 gate。
- 创建 `experiments/missing_m3_mosi_binary_task_20260829/results/`：轻量 config/history/metrics/prediction NPZ，不包含 checkpoint。

### 任务 1：锁定任务模式与监督语义

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [x] **步骤 1：编写失败的配置与 CLI 测试**

新增测试，要求默认回归、显式 binary 可解析，且非 MOSI 数据集拒绝 binary：

```python
def test_mosi_task_mode_defaults_to_regression_and_validates_binary_scope():
    required = [
        "--audio-feature", "a", "--text-feature", "t",
        "--video-feature", "v", "--output-dir", "out",
    ]
    assert build_parser().parse_args(required).mosi_task_mode == "regression"
    assert build_parser().parse_args(
        required + ["--dataset", "CMUMOSI", "--mosi-task-mode", "binary"]
    ).mosi_task_mode == "binary"
    with pytest.raises(ValueError, match="CMUMOSI"):
        train_gcnet._resolve_task_contract("IEMOCAPSix", "binary")
```

- [x] **步骤 2：运行测试验证失败**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py -k 'mosi_task_mode'
```

预期：FAIL，缺少 `mosi_task_mode` 或 `_resolve_task_contract`。

- [x] **步骤 3：实现最小任务 contract**

在 `TrainConfig` 末尾追加字段，保持旧 positional 顺序：

```python
mosi_task_mode: str = "regression"
```

新增：

```python
def _resolve_task_contract(dataset: str, mosi_task_mode: str) -> Dict[str, object]:
    shape = _dataset_shape(dataset)
    if mosi_task_mode not in {"regression", "binary"}:
        raise ValueError("mosi_task_mode must be 'regression' or 'binary'")
    if mosi_task_mode == "binary":
        if dataset != "CMUMOSI":
            raise ValueError("binary mosi_task_mode is only valid for CMUMOSI")
        shape.update(task="binary", num_classes=2)
    return shape
```

CLI 增加 `--mosi-task-mode`，choices 为 `regression/binary`，默认 `regression`；将值写入 `TrainConfig`、config JSON 和 checkpoint config。

- [x] **步骤 4：运行测试验证通过**

重复步骤 2；预期 PASS。

### 任务 2：实现无零 Binary CE 与可审计预测

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [x] **步骤 1：编写失败的 loss/collection 测试**

```python
def test_binary_mosi_ce_excludes_zero_and_padding_but_keeps_finite_graph_loss():
    logits = torch.tensor(
        [[[4.0, -1.0]], [[9.0, -9.0]], [[-2.0, 3.0]], [[1.0, 1.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-1.0, 0.0, 2.0, -3.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    loss = _task_loss("CMUMOSI", logits, labels, umask, "binary")
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 0], logits[2, 0]]), torch.tensor([0, 1])
    )
    torch.testing.assert_close(loss, expected)
    zero = _task_loss(
        "CMUMOSI", logits[:2], torch.zeros(1, 2), torch.ones(1, 2), "binary"
    )
    assert torch.isfinite(zero) and zero.item() == 0.0
    zero.backward()
    assert logits.grad is not None
```

预测收集测试要求只返回 nonzero utterances，同时保存原连续标签：

```python
predicted, binary, continuous = train_gcnet._collect_predictions(
    "CMUMOSI", logits, labels, umask, "binary"
)
assert predicted.tolist() == [0, 1]
assert binary.tolist() == [0, 1]
assert continuous.tolist() == [-1.0, 2.0]
```

- [x] **步骤 2：运行测试验证失败**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py -k 'binary_mosi_ce or binary_prediction'
```

预期：FAIL，helper 尚不接受 task mode 或返回三元素。

- [x] **步骤 3：实现 task-mode-aware helper**

`_task_loss(..., mosi_task_mode="regression")` 的 binary 分支：

```python
flat_logits = logits.transpose(0, 1).reshape(-1, 2)
continuous = labels.reshape(-1)
selected = umask.reshape(-1).bool() & continuous.ne(0)
if not bool(selected.any()):
    return flat_logits.sum() * 0.0
targets = continuous[selected].gt(0).long()
return torch.nn.functional.cross_entropy(flat_logits[selected], targets)
```

`_collect_predictions` 返回 `(predictions, metric_labels, continuous_labels)`。Binary collection 使用 `umask & labels.ne(0)`；regression collection 保留全部有效 utterance。

`_metrics(..., mosi_task_mode="regression")` 的 binary 输入已是 0/1，直接计算 accuracy、weighted F1、macro F1，不计算 MAE/correlation。

- [x] **步骤 4：更新 train/eval 调用点与 artifact**

`train_epoch`、`evaluate_rate` 传入 task mode。Binary NPZ 保存：

```python
{
    "predictions": predictions_array,
    "labels": binary_labels_array,
    "continuous_labels": continuous_labels_array,
    "availability": availability_for_nonzero_items,
}
```

artifact availability 与 supervised items 等长；`mask_sha256` 仍基于全部有效 utterance availability，以便与 regression control 配对。

- [x] **步骤 5：运行测试验证通过**

重复步骤 2；预期 PASS。

### 任务 3：模型输出维度与回归等价

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [x] **步骤 1：编写模型构造与回归兼容测试**

```python
def test_binary_mode_builds_two_class_head_and_regression_keeps_one_class():
    assert train_gcnet._resolve_task_contract("CMUMOSI", "regression")["num_classes"] == 1
    assert train_gcnet._resolve_task_contract("CMUMOSI", "binary")["num_classes"] == 2
```

扩展已有 regression tests，明确默认参数与显式 `regression` 的 loss、metrics、predictions 完全相同；旧 positional `TrainConfig` 仍得到 `regression`。

- [x] **步骤 2：运行测试验证失败**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py -k 'two_class_head or regression_keeps or positional'
```

预期：新 binary assertion FAIL，旧回归测试保持 PASS。

- [x] **步骤 3：让 run_experiment 使用 task contract**

```python
shape = _resolve_task_contract(
    config_value.dataset, config_value.mosi_task_mode
)
```

模型继续从 `shape["num_classes"]` 获取输出维度。IEMOCAP 与默认 regression 的初始化顺序不改变。

- [x] **步骤 4：运行完整回归测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q \
  tests/test_missing_m3.py tests/test_mosi_text_lora.py
```

预期：全部 PASS，原 41 tests 不减少。

- [x] **步骤 5：运行静态验证**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m py_compile \
  gcnet_missing_m3/train_gcnet.py tests/test_missing_m3.py
git diff --check
```

预期：退出码 0。

### 任务 4：远程集成、seed66 正式实验与 gate

**文件：**
- 创建：`experiments/missing_m3_mosi_binary_task_20260829/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_binary_task_20260829/results/formal/seed_66/`

- [ ] **步骤 1：同步并运行真实集成**

只同步修改的 source/test 到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev`。在 biggpu 的 `s0` 环境运行 focused tests；随后在 GCNet official 环境用真实 MOSI batch 验证 binary logits `[L,B,2]`、CE finite、zero exclusion 和 backward。

- [ ] **步骤 2：运行正式 seed66**

使用 GPU5/6/7 中空闲且非 GPU4 的一张卡：

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  -m gcnet_missing_m3.train_gcnet \
  --dataset CMUMOSI \
  --feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features \
  --audio-feature wav2vec-large-c-UTT \
  --text-feature deberta-large-4-UTT \
  --video-feature manet_UTT \
  --output-dir /data2/yb/remote_experiments/missing_m3_mosi_binary_task_20260829/formal/seed_66 \
  --seed 66 --fold 1 --epochs 100 --batch-size 32 \
  --train-rate-mode all --hidden 200 --fusion-type slot \
  --windowp 2 --windowf 2 --gradient-clip-norm 1.0 \
  --mosi-task-mode binary --device cuda
```

- [ ] **步骤 3：审计结果**

检查 history=100、validation-only checkpoint、8 NPZ 重算、continuous labels 无零、8 full mask SHA 与 paired regression 相同、parameter delta 只来自 classifier 新增一行、无单类别坍塌。

- [ ] **步骤 4：应用预注册 gate**

```text
miss0 >= 87.5
miss0 delta vs regression >= +1.0
nonzero mean delta vs regression >= -0.5
```

PASS 才创建 seeds 67--70；FAIL 则下一问题锁定为 MOSI 单说话人 Speaker branch，不修改 loss 或阈值。

- [ ] **步骤 5：记录、Lore commit 与推送**

报告明确这是 task-protocol alignment，不是结构贡献。提交记录实际测试、NPZ 审计和未扩展边界；推送 `feature/m3-jepa-gcnet` 到 GitHub。
