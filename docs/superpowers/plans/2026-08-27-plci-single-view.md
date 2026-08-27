# Single-View PLCI-JEPA 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 PLCI 使用官方 Natural mask、Natural student latent 和同一次 GCNet hidden 计算 JEPA loss，删除 Single-View 路径的平衡辅助采样与第二次 GCNet forward。

**架构：** 新增一个很薄的 `SingleViewPLCIJEPAGraphModel`，继承现有已测试的 PLCI 模型，只负责从 Natural view 中筛选不完整 utterance 并调用原 source-anchored predictor。共享训练器新增 `plci-single` 路由，Dual-View 的 `plci` 路由保持原行为；正式实验复用固定 mask bank 和已有 Original 结果。

**技术栈：** Python 3、PyTorch、PyTorch Geometric、pytest、现有 GCNet/PLCI 训练器、SSH `biggpu`。

---

## 文件结构

- 创建：`gcnet_plci_single_view/__init__.py`——导出 Single-View 模型。
- 创建：`gcnet_plci_single_view/model.py`——Natural target selection，不复制 PLCI 主体。
- 创建：`tests/test_plci_single_view_model.py`——一次 forward、ATV 跳过、泄漏和梯度测试。
- 修改：`gcnet_modality_jepa/train_gcnet.py`——新增 `plci-single` 路由及 Single-View loss 分支。
- 修改：`tests/test_plci_training.py`——CLI、构造、参数校验与 Dual-View 不回归测试。
- 创建：`scripts/run_plci_single_view_iemocap6.py`——只调度 0.0/0.5/0.7 × 5 seeds，继承 Original。
- 创建：`tests/test_plci_single_view_runner.py`——任务矩阵、固定 mask、禁止 Original 重跑。
- 创建：`docs/experiments/plci-single-view-stage1.md`——正式命令、输出根目录和停止/扩展规则。

### 任务 1：锁定 Natural target selection

**文件：**
- 创建：`tests/test_plci_single_view_model.py`
- 创建：`gcnet_plci_single_view/__init__.py`
- 创建：`gcnet_plci_single_view/model.py`

- [ ] **步骤 1：编写失败的 mixed-pattern 测试**

测试构造同时包含 `A`、`AT`、`TV` 和 `ATV` 的 conversation，调用：

```python
log_prob, reconstruction, hidden, latents = model.forward_natural(
    [masked], availability, qmask, umask, lengths
)
predictions = model.predict_natural(
    latents, hidden, availability, umask
)
```

断言：

```python
assert {record.utterance_index for record in predictions.targets} == {0, 1, 3}
assert all(record.utterance_index != 2 for record in predictions.targets)
assert model.last_prediction_umask[0, 2].item() == 0
```

- [ ] **步骤 2：运行测试并确认缺少模块**

运行：

```bash
pytest -q tests/test_plci_single_view_model.py
```

预期：collection FAIL，`ModuleNotFoundError: gcnet_plci_single_view`。

- [ ] **步骤 3：实现最小 Single-View 包装器**

`gcnet_plci_single_view/model.py` 的核心实现为：

```python
from gcnet_plci_jepa.model import PLCIJEPAGraphModel


class SingleViewPLCIJEPAGraphModel(PLCIJEPAGraphModel):
    def predict_natural(self, student_latents, hidden, availability, umask):
        valid = self._validate_availability(availability, umask, allow_atv=True)
        incomplete = availability.sum(dim=-1).lt(3) & valid
        prediction_umask = incomplete.T.to(dtype=umask.dtype)
        self.last_prediction_umask = prediction_umask.detach().clone()
        return self.predictor(
            student_latents,
            hidden,
            availability,
            prediction_umask,
        )
```

`gcnet_plci_single_view/__init__.py` 只导出该类。

- [ ] **步骤 4：补充全 ATV 和非法 pattern 测试**

全 ATV 必须满足：

```python
assert not predictions.targets
loss, counts = plci_jepa_loss(predictions, teacher_targets)
assert loss.item() == 0.0
assert counts == {
    "utterances": 0, "targets": 0, "paths": 0,
    "audio_targets": 0, "audio_paths": 0,
    "text_targets": 0, "text_paths": 0,
    "visual_targets": 0, "visual_paths": 0,
}
```

有效位置 `000` 必须抛出包含 `pattern` 的 `ValueError`。

- [ ] **步骤 5：运行模型单元测试**

运行：

```bash
pytest -q tests/test_plci_single_view_model.py tests/test_plci_model.py tests/test_plci_predictor.py tests/test_plci_modules.py
```

预期：全部 PASS，现有 Dual-View 测试不变。

- [ ] **步骤 6：提交模型边界**

提交必须使用 Lore trailers，并记录 `ATV` 只从 PLCI target selector 跳过、不能从 GCNet graph 跳过。

### 任务 2：将共享训练器接入一次 Natural forward

**文件：**
- 修改：`gcnet_modality_jepa/train_gcnet.py:56-63,280-304,572-598,848-872,997-1005,1097-1126,1552-1560,1784-1794`
- 修改：`tests/test_plci_training.py`

- [ ] **步骤 1：编写 CLI 与构造失败测试**

新增断言：

```python
args = build_argument_parser().parse_args([
    "--base-model", "LSTM",
    "--jepa-architecture", "plci-single",
])
assert args.jepa_architecture == "plci-single"
model = build_model(_args(jepa_architecture="plci-single", hidden=8), 2, 3, 4)
assert isinstance(model, SingleViewPLCIJEPAGraphModel)
```

同时验证 `plci` 仍构造 `PLCIJEPAGraphModel`。

- [ ] **步骤 2：运行测试确认 parser 拒绝新值**

运行：

```bash
pytest -q tests/test_plci_training.py::test_build_model_routes_single_view_plci_architecture
```

预期：FAIL，argument choice 或类型断言失败。

- [ ] **步骤 3：新增明确的架构判定函数**

在训练器中加入：

```python
def _plci_mode(args):
    architecture = getattr(args, "jepa_architecture", "independent")
    return architecture if architecture in {"plci", "plci-single"} else None
```

使用：

```python
plci_mode = _plci_mode(args)
plci_enabled = plci_mode is not None
plci_dual_view = plci_mode == "plci"
plci_single_view = plci_mode == "plci-single"
```

只有 `plci_dual_view` 可以创建或消费 `plci_aux_generator`。

- [ ] **步骤 4：复用统一模型参数构造**

选择类而不复制参数列表：

```python
model_class = (
    SingleViewPLCIJEPAGraphModel
    if architecture == "plci-single"
    else PLCIJEPAGraphModel
)
model = model_class(...existing PLCI keyword arguments...)
```

`validate_training_args` 对 `plci` 和 `plci-single` 执行同一组约束。

- [ ] **步骤 5：编写一次 hidden forward 的训练测试**

使用一个最小 dataloader batch 和 mock：

```python
with mock.patch.object(model, "encode_hidden", wraps=model.encode_hidden) as encode:
    train_or_eval_model(..., train=True, plci_aux_generator=None)
assert encode.call_count == 1
```

并 patch `sample_balanced_patterns`，断言 `assert_not_called()`。

- [ ] **步骤 6：实现 Single-View loss 分支**

Natural forward 保存 latents：

```python
log_prob, recon_input_features, hidden, natural_latents = model.forward_natural(...)
```

训练期 Single-View 使用：

```python
natural_predictions = model.predict_natural(
    natural_latents, hidden, input_features_mask[0], umask
)
with torch.no_grad():
    teacher_targets = model.encode_teacher_targets(input_features[0])
loss3, missing_counts = plci_jepa_loss(natural_predictions, teacher_targets)
```

Dual-View 分支保持原 `sample_balanced_patterns -> forward_auxiliary` 代码。

- [ ] **步骤 7：记录 Natural pattern counts**

Single-View 的 `diagnostics["plci"]["pattern_counts"]` 从：

```python
valid_natural = input_features_mask[0][umask.T.bool()]
```

统计六种 incomplete pattern；`ATV` 单独记录为 `111`，证明 rate 0
没有伪造 auxiliary target。

- [ ] **步骤 8：运行训练器测试**

运行：

```bash
pytest -q tests/test_plci_training.py tests/test_plci_single_view_model.py
```

预期：全部 PASS。

- [ ] **步骤 9：提交训练器路由**

Lore `Directive` 必须注明 `plci` 是 Dual-View、`plci-single` 是
Single-View，不能交换命名或默认行为。

### 任务 3：证明无泄漏、梯度和 Dual-View 不回归

**文件：**
- 修改：`tests/test_plci_single_view_model.py`
- 修改：`tests/test_plci_training.py`

- [ ] **步骤 1：编写 Natural target 泄漏测试**

构造同一 `masked`、`availability`、`qmask`、`umask`，只改变
`full_features` 中自然缺失的 target block。分别计算 student forward 和
teacher target，断言：

```python
ASSERT_CLOSE(hidden_1, hidden_2, rtol=0, atol=0)
for first, second in zip(predictions_1.targets, predictions_2.targets):
    ASSERT_CLOSE(first.paths, second.paths, rtol=0, atol=0)
assert not torch.equal(teacher_1[missing_name], teacher_2[missing_name])
```

- [ ] **步骤 2：编写梯度覆盖测试**

组合 classification、reconstruction 和 `plci_jepa_loss` 后 backward，断言
student projectors、predictor、Temporal GCNet 至少一个参数获得有限非零梯度，
teacher 全部 `grad is None`。

- [ ] **步骤 3：运行 PLCI 完整测试组一次**

运行：

```bash
pytest -q \
  tests/test_plci_single_view_model.py \
  tests/test_plci_model.py \
  tests/test_plci_modules.py \
  tests/test_plci_patterns.py \
  tests/test_plci_predictor.py \
  tests/test_plci_training.py
```

预期：全部 PASS。不要重复运行相同测试矩阵。

- [ ] **步骤 4：运行一次 CPU 集成 forward/backward**

运行一个 mixed-pattern batch，输出：

```text
encode_hidden_calls=1
jepa_targets>0
loss_finite=true
teacher_gradients=0
```

- [ ] **步骤 5：提交协议验证**

记录 exact pytest 命令和结果，不提交生成缓存。

### 任务 4：建立不重跑 Original 的 Stage-1 runner

**文件：**
- 创建：`scripts/run_plci_single_view_iemocap6.py`
- 创建：`tests/test_plci_single_view_runner.py`
- 创建：`docs/experiments/plci-single-view-stage1.md`

- [ ] **步骤 1：编写 15-task 矩阵测试**

Runner 暴露纯函数：

```python
jobs = build_jobs(
    rates=(0.0, 0.5, 0.7),
    seeds=(66, 67, 68, 69, 70),
    fold=5,
)
assert len(jobs) == 15
assert {job.method for job in jobs} == {"plci-single"}
assert all("--jepa-architecture" in job.command for job in jobs)
assert all("plci-single" in job.command for job in jobs)
```

任何 `original` 训练命令都使测试失败。

- [ ] **步骤 2：编写固定 mask 与碰撞测试**

每个 job 必须包含：

```text
--mask-bank-root <existing-fixed-bank>
--fold 5
--seed <66..70>
--mask-type constant-<rate>
--evaluation-protocol strict
```

输出目录编码 dataset、fold、rate、seed 和方法；已完成 manifest 继承，半写目录
拒绝覆盖。

- [ ] **步骤 3：实现最小 runner**

Runner 只负责生成命令、健康 GPU 分配、最多每卡三个任务以及状态汇总。它不生成
mask bank、不运行 Original、不解析半写 manifest 为完成结果。

- [ ] **步骤 4：运行 runner 测试一次**

运行：

```bash
pytest -q tests/test_plci_single_view_runner.py
```

预期：全部 PASS，15 个唯一任务，0 个 Original 任务。

- [ ] **步骤 5：提交正式 runner**

文档记录输出根目录、Python 环境、GPU 排布、继承 Original 根目录和停止命令。

### 任务 5：远程一次预检并启动正式判别实验

**文件：**
- 只更新：`docs/experiments/plci-single-view-stage1.md` 的实际运行记录
- 远程输出：`experiments/plci_single_view_iemocap6_stage1_20260827/`

- [ ] **步骤 1：同步分支到 biggpu**

同步代码时排除 datasets、checkpoints、实验输出和 Git 元数据；远程代码目录固定，
不覆盖其他方法目录。

- [ ] **步骤 2：运行唯一一次 GPU forward/backward 预检**

使用正式 Python 环境：

```text
/data2/yb/reproduction_envs/gcnet-official/bin/python
```

预期：一次 hidden forward、finite loss/gradient、无 CUDA dtype/device 错误。失败则只
修复根因并重跑同一预检，不新增 smoke matrix。

- [ ] **步骤 3：验证正式输入**

启动前检查：

- dataset=`IEMOCAPSix`；
- fold=`5`；
- mask bank checksum 与 Original manifest 一致；
- 15 个 PLCI-Single 任务；
- 0 个 Original 任务；
- GPU 仅使用健康且未被其他用户占用的卡。

- [ ] **步骤 4：启动并持续监控 15 个正式任务**

每张健康 GPU 最多三个实验。训练期间只读取进度和异常，不修改模型、mask、超参数
或选择规则。

- [ ] **步骤 5：汇总严格配对结果**

每个 rate 输出：PLCI-Single 五个 seed、Original 五个 inherited seed、均值、delta、
正向 seed 数、最佳 epoch 与坍塌检查。0.5/0.7 均通过 Stage-1 gate 后，才生成全
missing-rate 扩展计划。
