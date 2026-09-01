# Mask-Aware SAM Backbone 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在独立包中实现可直接读取现有 GCNet utterance 特征的 mask-aware SAM-style 主干，并以 CMU-MOSI miss=0 五种子、validation-loss 选模实验判断是否值得连接 Missing-M3。

**架构：** 三个模态分别投影并沿对话序列进行自注意力，再通过六条候选有向跨模态注意力路径形成交互轨迹，最后仅在有效轨迹上池化。第一阶段只训练回归头，不实例化 GCNet、SDR、Missing-M3 predictor 或 EMA teacher。

**技术栈：** Python 3.10、PyTorch、NumPy、scikit-learn、pytest；复用仓库现有 CMU-MOSI loader、官方 split 和 JSON/NPZ 结果约定。

---

## 文件职责

- 创建 `gcnet_missing_m3_sam_backbone/__init__.py`：公开稳定模型接口。
- 创建 `gcnet_missing_m3_sam_backbone/attention.py`：带显式 query/key availability 的安全注意力与轨迹池化。
- 创建 `gcnet_missing_m3_sam_backbone/model.py`：模态编码、六方向交互和情绪回归头。
- 创建 `gcnet_missing_m3_sam_backbone/train_mosi.py`：单个 seed 的 miss=0 正式训练、验证选模与一次测试。
- 创建 `gcnet_missing_m3_sam_backbone/run_mosi.py`：五种子 GPU 调度、继承 control、结果汇总。
- 创建 `gcnet_missing_m3_sam_backbone/tests/test_attention.py`：mask 与泄漏不变量。
- 创建 `gcnet_missing_m3_sam_backbone/tests/test_model.py`：结构、梯度、pattern 行为。
- 创建 `gcnet_missing_m3_sam_backbone/tests/test_train_mosi.py`：数据/指标/选模与产物协议。
- 创建 `gcnet_missing_m3_sam_backbone/tests/test_runner.py`：五种子任务矩阵和汇总。
- 修改 `docs/superpowers/plans/2026-09-01-sam-backbone.md`：执行时勾选步骤并记录验证证据。

### 任务 1：实现不会读取缺失 key/value 的注意力原语

**文件：**
- 创建：`gcnet_missing_m3_sam_backbone/attention.py`
- 创建：`gcnet_missing_m3_sam_backbone/tests/test_attention.py`

- [ ] **步骤 1：编写失败的 mask 与空 key 测试**

```python
def test_masked_key_value_cannot_change_output():
    layer = SafeDirectedAttention(8, 2, 0.0).eval()
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 1, 8)
    changed = key.clone()
    changed[1] = 10000.0
    qmask = torch.ones(3, 1, dtype=torch.bool)
    kmask = torch.tensor([[1], [0], [1]], dtype=torch.bool)
    left, _ = layer(query, key, qmask, kmask)
    right, _ = layer(query, changed, qmask, kmask)
    torch.testing.assert_close(left, right)


def test_conversation_without_keys_returns_zero_track():
    layer = SafeDirectedAttention(8, 2, 0.0).eval()
    output, valid = layer(
        torch.randn(2, 1, 8),
        torch.randn(2, 1, 8),
        torch.ones(2, 1, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.bool),
    )
    assert not valid.any()
    assert torch.equal(output, torch.zeros_like(output))
```

- [ ] **步骤 2：运行红灯**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_attention.py -q`

预期：收集失败，`gcnet_missing_m3_sam_backbone.attention` 尚不存在。

- [ ] **步骤 3：实现安全有向注意力与池化**

实现接口：

```python
class SafeDirectedAttention(nn.Module):
    def forward(
        self,
        query: Tensor,       # [L,B,D]
        key_value: Tensor,   # [L,B,D]
        query_valid: Tensor, # [L,B]
        key_valid: Tensor,   # [L,B]
    ) -> tuple[Tensor, Tensor]:
        """Return a zero invalid track and [L,B] track validity."""


class MaskedTrackPooling(nn.Module):
    def forward(
        self,
        tracks: Tensor,      # [L,B,K,D]
        track_valid: Tensor, # [L,B,K]
    ) -> tuple[Tensor, Tensor]:
        """Softmax only over valid K tracks; reject rows with no valid track."""
```

按 batch conversation 单独处理全空 key，避免 `MultiheadAttention` 在全 `-inf` 行产生 NaN。输出最后乘 `query_valid`，无效位置严格为零。

- [ ] **步骤 4：验证绿灯**

运行：同步骤 2。

预期：全部通过，包含输出有限、masked value 不变性、空 key、池化权重和为 1。

- [ ] **步骤 5：提交**

提交仅包含 attention 实现与测试，使用 Lore commit trailer 记录空 key 的处理不变量。

### 任务 2：实现精简 SAM conversation backbone

**文件：**
- 创建：`gcnet_missing_m3_sam_backbone/__init__.py`
- 创建：`gcnet_missing_m3_sam_backbone/model.py`
- 创建：`gcnet_missing_m3_sam_backbone/tests/test_model.py`

- [ ] **步骤 1：编写失败的结构和 pattern 测试**

```python
def test_complete_forward_shape_and_backward():
    model = MaskAwareSAMModel(4, 6, 8, width=12, heads=3, dropout=0.0)
    features = [torch.randn(5, 2, d) for d in (4, 6, 8)]
    availability = torch.ones(5, 2, 3)
    umask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]).float()
    prediction, hidden, audit = model(features, availability, umask)
    assert prediction.shape == (5, 2, 1)
    assert hidden.shape == (5, 2, 12)
    prediction.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_missing_local_feature_has_no_effect():
    torch.manual_seed(7)
    model = MaskAwareSAMModel(4, 6, 8, width=12, heads=3, dropout=0.0).eval()
    features = [torch.randn(3, 1, d) for d in (4, 6, 8)]
    availability = torch.ones(3, 1, 3)
    availability[1, 0, 1] = 0
    changed = [value.clone() for value in features]
    changed[1][1, 0] = 10000.0
    umask = torch.ones(1, 3)
    left = model(features, availability, umask)[0]
    right = model(changed, availability, umask)[0]
    torch.testing.assert_close(left, right)
```

测试还要锁定 A/T/V singleton、AT/AV/TV、ATV、padding、全空有效 utterance 拒绝和邻居 observed key 可用。

- [ ] **步骤 2：运行红灯**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_model.py -q`

预期：`MaskAwareSAMModel` 导入失败。

- [ ] **步骤 3：实现模型**

稳定接口：

```python
class MaskAwareSAMModel(nn.Module):
    DIRECTIONS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))

    def forward(self, features, availability, umask):
        valid = umask.transpose(0, 1).bool()
        modality_valid = availability.bool() & valid.unsqueeze(-1)
        encoded = [
            encoder(value, modality_valid[..., index])
            for index, (encoder, value) in enumerate(zip(self.encoders, features))
        ]
        tracks = list(encoded)
        track_valid = [modality_valid[..., index] for index in range(3)]
        attention_maps = {}
        for source, target in self.DIRECTIONS:
            output, active, weights = self.cross_attention[
                "{}-{}".format(source, target)
            ](
                encoded[source],
                encoded[target],
                modality_valid[..., source],
                modality_valid[..., target],
                return_weights=True,
            )
            tracks.append(output)
            track_valid.append(active)
            attention_maps[(source, target)] = weights
        hidden, pooling = self.pool(
            torch.stack(tracks, dim=2),
            torch.stack(track_valid, dim=2),
        )
        hidden = self.output_block(hidden) * valid.unsqueeze(-1)
        prediction = self.regressor(hidden) * valid.unsqueeze(-1)
        return prediction, hidden, {
            "cross_attention": attention_maps,
            "track_pooling": pooling,
        }
```

每个 modality encoder 为 `LayerNorm -> Linear -> GELU -> Dropout -> one pre-norm TransformerEncoderLayer`。缺失输入在投影前后都乘 availability。三条 unimodal track 与六条 directed track 一起进入 `MaskedTrackPooling`。回归头为 `LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)`。

- [ ] **步骤 4：运行模型与联合测试**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_attention.py gcnet_missing_m3_sam_backbone/tests/test_model.py -q`

预期：全部通过。

- [ ] **步骤 5：提交**

提交模型、公开接口和测试；Directive 明确 missing feature 不得进入投影与 K/V。

### 任务 3：实现单 seed MOSI miss=0 训练协议

**文件：**
- 创建：`gcnet_missing_m3_sam_backbone/train_mosi.py`
- 创建：`gcnet_missing_m3_sam_backbone/tests/test_train_mosi.py`

- [ ] **步骤 1：编写失败的数据、指标和选模测试**

```python
def test_nonzero_metrics_match_existing_implementation():
    labels = np.array([-1.0, 0.0, 1.0, 2.0])
    predictions = np.array([-0.2, -1.0, 0.1, -0.3])
    metrics = regression_metrics(labels, predictions)
    assert metrics["sample_count"] == 3


def test_best_epoch_is_selected_only_by_validation_loss():
    records = [
        {"epoch": 1, "validation": {"loss": 0.5}, "test": {"weighted_f1": 0.9}},
        {"epoch": 2, "validation": {"loss": 0.4}, "test": {"weighted_f1": 0.7}},
    ]
    assert select_best_epoch(records) == 2
```

再测试模型只收到 complete features、availability 全一、test loader 不在训练 epoch 内调用，以及输出原子写入。

- [ ] **步骤 2：运行红灯**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_train_mosi.py -q`

预期：训练模块不存在。

- [ ] **步骤 3：实现训练入口**

定义不可变配置：

```python
@dataclass(frozen=True)
class SAMTrainConfig:
    seed: int = 66
    width: int = 120
    heads: int = 4
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.1
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
```

复用 `gcnet_modality_jepa.train_gcnet.get_loaders`、`SeedBundle` 和数据 batch 格式。每个 epoch 只计算 train 与 validation。依据最小 validation MSE 保存一个 best state；训练结束后加载 best state，仅测试一次。保存：

```text
config.json
history.json
best_checkpoint.pt
metrics.json
predictions.npz
```

`metrics.json` 必须记录 best epoch、validation loss、associated test Acc-2/W-F1/MAE、参数量、峰值显存、运行时间和选择 split。

- [ ] **步骤 4：运行训练协议测试与一轮 CPU/单 GPU smoke**

运行：

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_train_mosi.py -q
/data2/yb/reproduction_envs/gcnet-official/bin/python -m gcnet_missing_m3_sam_backbone.train_mosi --epochs 1 --device cuda:0 --seed 66 --output-dir /tmp/sam_mosi_smoke --feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features
```

预期：测试通过；smoke 生成五个产物，所有指标有限，`selection_split=validation`。

- [ ] **步骤 5：提交**

提交训练入口和测试；Not-tested 记录尚未执行五种子正式训练。

### 任务 4：实现五种子调度、继承控制和门控汇总

**文件：**
- 创建：`gcnet_missing_m3_sam_backbone/run_mosi.py`
- 创建：`gcnet_missing_m3_sam_backbone/tests/test_runner.py`

- [ ] **步骤 1：编写失败的 job matrix 与汇总测试**

```python
def test_jobs_are_exactly_five_seeds_without_control_reruns(tmp_path):
    jobs = build_jobs(tmp_path, gpus=(0, 1, 2))
    assert [job.seed for job in jobs] == [66, 67, 68, 69, 70]
    assert all("Original" not in job.command for job in jobs)


def test_gate_requires_mean_and_three_paired_wins():
    summary = summarize(candidate, inherited_control)
    assert summary["passed"] == (
        summary["mean_delta"] > 0 and summary["positive_seed_count"] >= 3
    )
```

- [ ] **步骤 2：运行红灯**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_runner.py -q`

预期：runner 模块不存在。

- [ ] **步骤 3：实现 runner**

Runner 接受 `--gpus 0,1,2`，以轮询方式将五个 seed 分配到 GPU，每张卡同时最多一个训练进程。它继承并校验当前仓库 strongest miss0 control 的五种子结果，不启动 control 子进程。任务退出后验证五个产物、prediction 非坍塌和 selection provenance，原子写入 `summary.json` 与 `summary.md`。

- [ ] **步骤 4：运行 runner 测试**

运行：`/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests/test_runner.py -q`

预期：全部通过，模拟失败进程可被回收，半写 metrics 不会被继承。

- [ ] **步骤 5：提交**

提交 runner 与测试；Directive 写明不得重跑 Original 或改用 test-oracle。

### 任务 5：正式验证并执行 MOSI miss=0 五种子

**文件：**
- 修改：`docs/superpowers/plans/2026-09-01-sam-backbone.md`
- 生成：`gcnet_missing_m3_sam_backbone/results/formal/*`（结果文件，不提交 checkpoint）

- [ ] **步骤 1：运行完整本地测试**

运行：

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest gcnet_missing_m3_sam_backbone/tests -q
git diff --check
```

预期：全部测试通过且无 whitespace error。

- [ ] **步骤 2：执行一次 GPU 前后向与一 epoch smoke**

只运行任务 3 的 smoke 命令一次；若已经生成且 provenance 匹配则继承，不重复执行。

- [ ] **步骤 3：启动正式五种子**

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m gcnet_missing_m3_sam_backbone.run_mosi \
  --gpus 0,1,2 \
  --output-root gcnet_missing_m3_sam_backbone/results/formal
```

预期：五个 candidate job，零个 Original job。

- [ ] **步骤 4：读取并审计结果**

确认 `summary.json` 中：五个 seed 完整、validation-loss 选模、test 只与 best validation epoch 关联、没有 one-class/constant/NaN、参数量与运行时间齐全。

- [ ] **步骤 5：应用门控**

- PASS：另建 Missing-M3 integration 规格，不在本计划内直接追加代码。
- FAIL：记录负结果并关闭 SAM backbone，不运行 missing rates。

- [ ] **步骤 6：最终提交**

提交代码、测试、`summary.json`、`summary.md` 和更新后的计划；不提交 `best_checkpoint.pt`、大体积 NPZ 或用户已有未跟踪实验目录。

## 计划自检

- 规格中的 architecture、mask 不变量、Stage 1 门控、测试、独立目录和 non-goals 均有对应任务。
- 接口名称在任务间一致：`SafeDirectedAttention`、`MaskedTrackPooling`、`MaskAwareSAMModel`、`SAMTrainConfig`。
- 第一阶段没有 Missing-M3、EMA、重建、GCNet graph、SAM-LML 额外辅助损失或 Original 重跑。
- 正式选模只依赖 validation loss；test 不参与 epoch 选择。
