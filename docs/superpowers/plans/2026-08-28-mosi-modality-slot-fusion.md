# MOSI Modality-Slot Fusion 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Missing-M3 GCNet 增加向后兼容的 modality-slot observed-set fusion，并在 CMU-MOSI 上完成五种子混合 missing-rate 判别实验。

**架构：** 现有 `mean` 路径保持不变；新 `slot` 路径将三个带身份的 observed latent 与一个 pattern embedding 拼接，再映射回原 latent width。其余模型与实验协议不变，旧 mean 结果作为继承对照。

**技术栈：** Python、PyTorch、pytest、GCNet、远程 V100、JSON/NPZ 实验制品。

---

## 文件职责

- 修改 `tests/test_missing_m3.py`：锁定 slot 身份、无泄漏、padding 和 CLI 行为。
- 修改 `gcnet_missing_m3/model.py`：实现 `mean|slot` 两种融合并向模型构造器传参。
- 修改 `gcnet_missing_m3/train_gcnet.py`：配置、CLI、checkpoint provenance 记录融合类型。
- 创建 `experiments/missing_m3_mosi_slot_20260828/EXPERIMENT.md`：登记命令、环境、运行状态和最终结果。
- 创建 `experiments/missing_m3_mosi_slot_20260828/results/`：保存从远程拉回的正式指标和预测制品。

### 任务 1：以失败测试锁定 Slot Fusion 合同

- [ ] **步骤 1：编写失败测试**

在 `tests/test_missing_m3.py` 中增加：

```python
def test_slot_fusion_preserves_modality_slots_without_missing_value_leakage():
    encoder = ObservedSetEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0, fusion_type="slot"
    ).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0
    first, _ = encoder(features, availability, umask)
    second, _ = encoder(changed, availability, umask)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)
```

再增加 `fusion_type="invalid"` 抛出 `ValueError`、slot padding 为零、parser 接受 `--fusion-type slot` 的独立测试。

- [ ] **步骤 2：确认测试因功能缺失而失败**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'slot or fusion_type'
```

预期：`ObservedSetEncoder.__init__()` 不接受 `fusion_type`，测试失败。

### 任务 2：实现最小 Slot Fusion

- [ ] **步骤 1：修改模型**

在 `gcnet_missing_m3/model.py` 中验证 `fusion_type in {"mean", "slot"}`。`mean` 继续构造输入宽度为 `d` 的原 `self.fusion`；`slot` 构造输入宽度为 `4d` 的 `self.fusion`。forward 中：

```python
if self.fusion_type == "mean":
    fused_input = evidence / count + pattern
else:
    fused_input = torch.cat([audio_slot, text_slot, visual_slot, pattern], dim=-1)
node[valid] = self.fusion(fused_input[valid])
```

三个 slot 只在对应 availability 为 1 时写入 `projected + modality_embedding`。

- [ ] **步骤 2：逐层传递配置**

为 `MissingM3GraphModel`、`TrainConfig` 增加 `fusion_type="mean"`，模型创建时传递；CLI 增加：

```python
parser.add_argument("--fusion-type", choices=("mean", "slot"), default="mean")
```

- [ ] **步骤 3：确认新旧测试全部通过**

运行：

```bash
pytest -q tests/test_missing_m3.py
```

预期：所有测试通过，现有 mean 测试无回归。

### 任务 3：验证兼容性和远程运行

- [ ] **步骤 1：比较 mean 旧路径**

在同一随机种子下构造默认 encoder 与显式 `fusion_type="mean"` encoder，复制 state dict 后比较七 pattern 输出，要求 `rtol=0, atol=0`。

- [ ] **步骤 2：远程检查环境与 GPU**

通过 `ssh biggpu` 检查官方 Python、代码目录、GPU 0/1/2/3/5 和已有进程；不使用 GPU 4。

- [ ] **步骤 3：同步并运行 GPU smoke**

同步本分支到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev`，使用官方环境运行 `fusion_type=slot` 的 MOSI 短前向/反向，要求 loss、gradient、输出均有限。

### 任务 4：运行 MOSI 五种子正式实验

- [ ] **步骤 1：建立唯一 manifest**

每个 seed 只建立一个任务，命令显式包含：

```text
--dataset CMUMOSI --fusion-type slot --seed <66..70> --epochs 100
```

输出目录互不重叠，且记录 Git commit、Python、CUDA、GPU、开始时间和完整参数。

- [ ] **步骤 2：并行运行五个 seed**

分别使用 GPU 0、1、2、3、5。启动后检查每个任务至少完成一个 epoch 且日志持续更新；不运行 Original/mean。

- [ ] **步骤 3：收集结果**

确认五个 `metrics.json`、40 个 rate prediction NPZ、checkpoint 和日志完整。拉回指标与预测制品，checkpoint 保留远程路径并记录 SHA256。

### 任务 5：分析、记录和提交

- [ ] **步骤 1：计算配对汇总**

对每个 rate 计算 slot 的五种子 W-F1 mean/std、MAE、correlation，并与既有 mean 结果按 seed/rate 对齐计算 delta 和正向 seed 数。

- [ ] **步骤 2：写实验报告**

更新 `EXPERIMENT.md`，明确 mean 是继承结果、slot 是本次新运行；不把非配对或单 seed 提升描述为稳定改善。

- [ ] **步骤 3：完整验证并提交推送**

运行单元测试、结果完整性检查与 `git diff --check`。使用 Lore commit message 提交代码和结果，然后推送 `feature/m3-jepa-gcnet`。

