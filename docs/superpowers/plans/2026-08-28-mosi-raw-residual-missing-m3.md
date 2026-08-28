# MOSI Raw-Residual Missing-M3 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 取消分类路径的 2560D→256D 提前压缩，以零初始化 modality residual 将 Student latent 接入原始不完整 GCNet，并验证 MOSI miss=0 五种子均值能否达到 88。

**架构：** 新增 `RawResidualObservedEncoder`，输出原始宽度的 observed raw feature 加 Student residual，同时输出现有 predictor 所需的 modality latents。`fusion_type=raw-residual` 时保留 `GraphModel` 原始 recurrent input；mean/slot 路径及所有训练协议不变。

**技术栈：** Python 3.8、PyTorch、PyTorch Geometric、pytest、GCNet、远程 V100、JSON/NPZ。

---

## 文件职责

- 修改 `tests/test_missing_m3.py`：锁定 raw preservation、无泄漏、recurrent width、梯度和 CLI 行为。
- 修改 `gcnet_missing_m3/model.py`：实现 `RawResidualObservedEncoder` 与模型路由。
- 修改 `gcnet_missing_m3/train_gcnet.py`：开放 `raw-residual` CLI/config。
- 创建 `experiments/missing_m3_mosi_raw_residual_20260828/EXPERIMENT.md`：记录 smoke、正式命令、结果与结论。
- 创建 `experiments/missing_m3_mosi_raw_residual_20260828/results/`：保存五种子轻量结果。

### 任务 1：用失败测试锁定 Raw-Residual Encoder

**文件：**
- 修改：`tests/test_missing_m3.py`

- [ ] **步骤 1：编写零初始化与无泄漏测试**

```python
def test_raw_residual_encoder_starts_as_exact_masked_raw_input():
    encoder = RawResidualObservedEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0
    ).eval()
    availability = _all_patterns()
    features = torch.randn(7, 1, 9)
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    output, latents = encoder(features, availability, torch.ones(1, 7))
    ASSERT_CLOSE(output, features * expanded, rtol=0, atol=0)
    assert all(value.shape == (7, 1, 8) for value in latents.values())
```

再构造 `changed[expanded == 0] += 10000`，要求 output 与所有 latents 不变；增加 padding 全零断言。

- [ ] **步骤 2：运行并确认正确红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q tests/test_missing_m3.py -k raw_residual
```

预期：collection 因 `RawResidualObservedEncoder` 尚不存在而失败。

### 任务 2：实现最小 RawResidualObservedEncoder

**文件：**
- 修改：`gcnet_missing_m3/model.py`
- 测试：`tests/test_missing_m3.py`

- [ ] **步骤 1：复用输入校验**

提取 module-level helper：

```python
def _validate_observed_inputs(features, availability, umask, dimensions):
    ...
    return valid
```

`ObservedSetEncoder._validate()` 委托该 helper，保持现有错误语义。

- [ ] **步骤 2：实现 residual adapters**

```python
self.adapters = nn.ModuleDict({
    name: nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, width))
    for name, width in zip(MODALITIES, self.dimensions)
})
for adapter in self.adapters.values():
    nn.init.zeros_(adapter[-1].weight)
    nn.init.zeros_(adapter[-1].bias)
```

forward 只对 `selected` observed utterances调用 projector/adapter；output 初始为零并 scatter `block + residual`。

- [ ] **步骤 3：运行 targeted tests**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q tests/test_missing_m3.py -k raw_residual
```

预期：encoder shape、exact masked raw、missing leakage、padding 测试全部通过。

### 任务 3：路由模型与 CLI

**文件：**
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 测试：`tests/test_missing_m3.py`

- [ ] **步骤 1：先写模型宽度与 CLI 失败测试**

```python
raw = MissingM3GraphModel(**_model_arguments(), fusion_type="raw-residual")
slot = MissingM3GraphModel(**_model_arguments(), fusion_type="slot")
assert raw.lstm.input_size == 9
assert slot.lstm.input_size == 8
```

parser 测试要求 `--fusion-type raw-residual` 被接受并写入 `TrainConfig`。

- [ ] **步骤 2：运行并确认红灯**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q tests/test_missing_m3.py -k 'recurrent_width or fusion_type'
```

预期：CLI choices 或模型路由拒绝 `raw-residual`。

- [ ] **步骤 3：实现模型路由**

```python
if fusion_type == "raw-residual":
    self.observed_set = RawResidualObservedEncoder(...)
else:
    self.observed_set = ObservedSetEncoder(..., fusion_type=fusion_type)
    # only here replace LSTM/GRU input with latent_dim
```

CLI choices 改为 `("mean", "slot", "raw-residual")`；默认仍为 `mean`。

- [ ] **步骤 4：验证完整模型 forward/backward**

使用七 pattern、真实 padding 和 `predict_missing=True`，确认 logits/hidden/prediction shape，loss backward 后 Student、adapter、GCNet 和 predictor 至少各有一个有限非零 gradient。

- [ ] **步骤 5：运行全部相关测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q tests/test_missing_m3.py
```

预期：全部通过，mean/slot 无回归。

### 任务 4：远程 MOSI Smoke 与实现提交

**文件：**
- 创建：`experiments/missing_m3_mosi_raw_residual_20260828/EXPERIMENT.md`

- [ ] **步骤 1：同步精确文件路径**

分别同步 `gcnet_missing_m3/model.py`、`gcnet_missing_m3/train_gcnet.py` 和 `tests/test_missing_m3.py` 到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev` 的对应路径；禁止多源 rsync 到仓库根目录。

- [ ] **步骤 2：运行远程测试与 1-epoch smoke**

```text
--dataset CMUMOSI --fusion-type raw-residual --seed 66 --epochs 1
```

要求真实 MOSI batch 完成 forward/backward/optimizer/EMA/checkpoint/八 rate test，无 NaN。

- [ ] **步骤 3：记录参数量与 smoke 指标并提交**

使用 Lore commit，明确本提交只改变 online input construction，正式结果尚未产生。

### 任务 5：五种子正式实验

**文件：**
- 更新：`experiments/missing_m3_mosi_raw_residual_20260828/EXPERIMENT.md`

- [ ] **步骤 1：启动唯一五任务矩阵**

Seeds/GPU：66/0、67/1、68/2、69/3、70/5。每个命令显式包含：

```text
--dataset CMUMOSI --fusion-type raw-residual --epochs 100
```

不运行 mean、slot 或 Original。

- [ ] **步骤 2：确认启动健康**

每个日志至少出现一个 epoch，GPU 4 保持空闲；若某任务失败，只诊断该任务，不重跑已完成 seed。

- [ ] **步骤 3：收集轻量制品**

拉回五份 config/history/metrics 与 40 个 prediction NPZ，checkpoint 留在 biggpu并记录 SHA256。

### 任务 6：严格评估 88 分门槛

**文件：**
- 更新：`experiments/missing_m3_mosi_raw_residual_20260828/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_raw_residual_20260828/results/SUMMARY.json`

- [ ] **步骤 1：完整性与配对审计**

重算 40 个 NPZ 的 Acc-2/W-F1/MAE/correlation；核对 40/40 mask SHA256 与 Slot/Mean 同 seed-rate 一致；确认 5×100 histories 有限。

- [ ] **步骤 2：计算 Primary/Secondary**

```text
Primary: miss0 five-seed mean W-F1 >= 88.0
Secondary: nonzero-rate mean >= slot nonzero-rate mean - 0.5
Seed gate: >=3/5 miss0 seeds beat paired Slot
```

- [ ] **步骤 3：按事实写结论**

通过 Primary 才能标记 `candidate_pass`；否则标记 `failed_target_88`，并记录 raw path、adapter norm、validation trajectory 供下一迭代定位。

### 任务 7：审查、验证与推送

**文件：**
- 修改：上述代码、测试、报告与结果文件

- [ ] **步骤 1：请求独立代码/结果审查**

审查 mean/slot backward compatibility、raw leakage、参数路由、指标归因与 provenance；修复全部 Critical/High/Medium。

- [ ] **步骤 2：运行最终验证**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest -q tests/test_missing_m3.py
git diff --check
```

同时重新执行 40 NPZ、40 mask hash、五 checkpoint hash 审计。

- [ ] **步骤 3：Lore commit 并推送**

提交到 `feature/m3-jepa-gcnet`，推送 `github` remote；核对本地 HEAD 与远程 branch SHA 完全一致。

