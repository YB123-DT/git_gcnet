# CMU-MOSI 单说话人 Graph Branch 消融实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加参数与 RNG 保持的 `temporal-only` / `speaker-only` 路由，并完成 CMU-MOSI 五种子八 missing-rate 正式消融。

**架构：** Shared `GraphModel` 始终实例化两个原始 graph branch，仅在 `encode_hidden()` 中按 mode 选择执行路径。Missing-M3 只负责配置透传；默认 `both` 必须保持旧参数、state keys、RNG 和 forward。Original 五种子直接继承，新运行两个模式各五种子。

**技术栈：** Python、PyTorch、PyTorch Geometric、pytest、biggpu V100。

---

## 文件结构

- 修改 `gcnet_modality_jepa/model.py`：共享 graph branch 路由与 mode 校验。
- 修改 `gcnet_missing_m3/model.py`：Missing-M3 构造参数透传。
- 修改 `gcnet_missing_m3/train_gcnet.py`：TrainConfig、CLI、正式模型构造透传。
- 修改 `tests/test_plci_model.py`：共享 GraphModel 路由、参数与执行路径测试。
- 修改 `tests/test_missing_m3.py`：Missing-M3、CLI/config 与 backward 测试。
- 创建 `experiments/missing_m3_mosi_graph_branch_ablation_20260829/`：协议、轻量结果和汇总。

### 任务 1：共享 GraphModel 的分支路由

**文件：**
- 修改：`tests/test_plci_model.py`
- 修改：`gcnet_modality_jepa/model.py`

- [ ] **步骤 1：编写失败测试**

构造三个相同 state dict 的 `GraphModel`，要求：

```python
both = GraphModel(**args, graph_branch_mode="both").eval()
temporal = GraphModel(**args, graph_branch_mode="temporal-only").eval()
speaker = GraphModel(**args, graph_branch_mode="speaker-only").eval()
temporal.load_state_dict(both.state_dict(), strict=True)
speaker.load_state_dict(both.state_dict(), strict=True)

assert set(both.state_dict()) == set(temporal.state_dict()) == set(speaker.state_dict())
assert sum(p.numel() for p in both.parameters()) == sum(p.numel() for p in temporal.parameters())
torch.testing.assert_close(
    both.encode_hidden(inputs, qmask, umask, lengths),
    temporal.encode_hidden(inputs, qmask, umask, lengths)
    + speaker.encode_hidden(inputs, qmask, umask, lengths),
)
```

为未选分支注册会抛错的 pre-hook，确认其不执行；非法 mode 必须抛出 `ValueError`。

- [ ] **步骤 2：验证 RED**

```bash
pytest -q tests/test_plci_model.py -k graph_branch_mode
```

预期：FAIL，构造函数尚不接受 `graph_branch_mode`。

- [ ] **步骤 3：最小实现**

在 `GraphModel.__init__()` 末尾增加：

```python
graph_branch_mode="both"
```

校验并保存属性。`encode_hidden()` 中：

```python
hidden_temporal = None
hidden_speaker = None
if self.graph_branch_mode in {"both", "temporal-only"}:
    hidden_temporal = ...
if self.graph_branch_mode in {"both", "speaker-only"}:
    hidden_speaker = ...
if self.graph_branch_mode == "both":
    return hidden_temporal + hidden_speaker
if self.graph_branch_mode == "temporal-only":
    return hidden_temporal
return hidden_speaker
```

两个 graph modules 的构造顺序保持不变。

- [ ] **步骤 4：验证 GREEN 与共享回归**

```bash
pytest -q tests/test_plci_model.py
git diff --check
```

预期：全部通过。

### 任务 2：Missing-M3 配置透传与 backward

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

要求 CLI 默认 `both` 并接受两个单分支值；`main()` 捕获的 `TrainConfig` 必须保存显式值。模型测试要求：

```python
model = MissingM3GraphModel(
    **args,
    graph_branch_mode="temporal-only",
)
loss = model(...)[0].sum()
loss.backward()
assert any(p.grad is not None for p in model.graph_net_temporal.parameters())
assert all(p.grad is None for p in model.graph_net_speaker.parameters())
```

反向测试再对 `speaker-only` 对称执行。默认与显式 `both` 复制 state 后输出精确相同。

- [ ] **步骤 2：验证 RED**

```bash
pytest -q tests/test_missing_m3.py -k graph_branch_mode
```

预期：FAIL，Missing-M3/CLI 尚无该参数。

- [ ] **步骤 3：最小透传**

- `MissingM3GraphModel.__init__()` 末尾追加 `graph_branch_mode="both"` 并传入 `super()`；
- `TrainConfig` 末尾追加 `graph_branch_mode: str = "both"`；
- CLI 增加 `--graph-branch-mode`，choices 为三种模式；
- `main()` 与 `run_experiment()` 透传该字段。

- [ ] **步骤 4：完整验证**

```bash
pytest -q tests/test_plci_model.py tests/test_missing_m3.py tests/test_mosi_text_lora.py
python -m py_compile gcnet_modality_jepa/model.py gcnet_missing_m3/model.py gcnet_missing_m3/train_gcnet.py
git diff --check
```

预期：全部通过；旧测试数量不减少。

### 任务 3：远程集成与十个正式任务

**文件：**
- 创建：`experiments/missing_m3_mosi_graph_branch_ablation_20260829/results/formal/`

- [ ] **步骤 1：同步与远程测试**

只同步三个 source 与两个 test 文件到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev`。在 biggpu `s0` 环境运行 branch-focused tests；在 official 环境运行三种 mode 的真实 forward/backward。

- [ ] **步骤 2：正式运行**

共十个任务：

```text
temporal-only × seeds 66,67,68,69,70
speaker-only  × seeds 66,67,68,69,70
```

每个任务使用：CMUMOSI、Regression、Slot、hidden 200、window 2/2、100 epochs、train-rate-mode all。并行使用除 GPU4 外的空闲 GPU；一个任务结束后自动接续同卡下一个任务。

- [ ] **步骤 3：结果审计**

每个任务要求：history=100、8 NPZ 可重算、8 mask SHA 与同 seed Original 相同、无 NaN/坍塌、参数量与 Original 完全相同、无 checkpoint 进入 Git。

### 任务 4：汇总、判读与推送

**文件：**
- 创建：`experiments/missing_m3_mosi_graph_branch_ablation_20260829/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_graph_branch_ablation_20260829/results/SUMMARY.json`

- [ ] **步骤 1：计算五种子统计**

逐 rate 计算 mean、sample std、相对 Original paired delta；同时汇总 miss0 和 nonzero-rate mean。

- [ ] **步骤 2：按设计判读**

只允许三种结论：Speaker 干扰、直接相加问题、或双分支互补。不得根据结果新增权重、阈值或第三种模型。

- [ ] **步骤 3：完成前验证与上传**

重新运行完整测试、10×8 NPZ 审计和 80/80 mask 配对；Lore commit 后推送 `github/feature/m3-jepa-gcnet`，核对远端 SHA。
