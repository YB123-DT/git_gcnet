# MOSI 等参数 SDT-style 对话主干诊断实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> superpowers-zh:subagent-driven-development（推荐）或
> superpowers-zh:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法
> 跟踪进度。

**目标：** 在独立目录中实现 active-forward 参数近似匹配的全上下文 Transformer，
替换 Single-View Missing-M3 的 GCNet 对话主干，并完成 MOSI 五种子八 rate 正式诊断。

**架构：** 现有 Slot Observed-Set Encoder 输出 `[L,B,256]` 后，进入
`256→384` 投影、正弦位置编码、speaker embedding、5 层 Pre-LN Transformer 和
`384→250` 输出投影。分类头、Missing-M3 Predictor、EMA Teacher、mask、loss、训练
协议与 control 共用现有实现。

**技术栈：** Python、PyTorch、pytest、现有 GCNet/Missing-M3 训练器、
compute-helper、3 张 V100、Git/GitHub。

---

## 文件结构

**创建：**

- `gcnet_missing_m3_sdt_backbone/__init__.py`：公开新模型与锁定常量。
- `gcnet_missing_m3_sdt_backbone/model.py`：位置编码、对话 Transformer 与
  `MissingM3SDTModel`。
- `gcnet_missing_m3_sdt_backbone/train_gcnet.py`：独立模型构建、训练入口和结果
  provenance；复用现有训练/评估函数。
- `gcnet_missing_m3_sdt_backbone/run_mosi.py`：五种子 GPU runner、resume 与 manifest。
- `gcnet_missing_m3_sdt_backbone/README.md`：模型边界、运行方式和非创新声明。
- `gcnet_missing_m3_sdt_backbone/STATUS.md`：实验状态。
- `gcnet_missing_m3_sdt_backbone/tests/test_model.py`：主干和集成合同测试。
- `gcnet_missing_m3_sdt_backbone/tests/test_train_gcnet.py`：构建、CLI、provenance 测试。
- `gcnet_missing_m3_sdt_backbone/tests/test_runner.py`：dry-run、任务数和 resume 测试。
- `gcnet_missing_m3_sdt_backbone/results/README.md`：结果目录约束。

**不修改：**

- `gcnet_missing_m3/model.py`
- `gcnet_missing_m3/loss.py`
- `gcnet_missing_m3/mixed_rate.py`
- `gcnet_missing_m3/train_gcnet.py`

## 任务 1：锁定模型接口的失败测试

**文件：**

- 创建：`gcnet_missing_m3_sdt_backbone/__init__.py`
- 创建：`gcnet_missing_m3_sdt_backbone/tests/__init__.py`
- 创建：`gcnet_missing_m3_sdt_backbone/tests/test_model.py`

- [ ] **步骤 1：创建最小包入口**

```python
"""Equal-active-budget SDT-style diagnostic for Missing-M3."""
```

- [ ] **步骤 2：编写形状、padding 与 full-context 失败测试**

测试固定使用 `input_dim=256`、`output_dim=250`、`d_model=384`、`ff_dim=704`、
`heads=8`、`layers=5`、`dropout=0.0`。核心断言：

```python
hidden = backbone(values, qmask, umask, [4, 2])
assert hidden.shape == (4, 2, 250)
assert torch.equal(hidden[2:, 1], torch.zeros_like(hidden[2:, 1]))

changed = values.clone()
changed[3, 0] += 2.0
hidden_changed = backbone(changed, qmask, umask, [4, 2])
assert not torch.allclose(hidden[0, 0], hidden_changed[0, 0])
```

另构造只改变 padding values 的输入，要求全部有效位置输出精确相同。

- [ ] **步骤 3：编写输入合同失败测试**

逐一验证：非二值 `umask`、非连续有效前缀、错误 `seq_lengths`、有效 utterance 的
speaker-ID `qmask [B,L]` 含负数、越界值、非整数或 NaN、长度超过 512，均抛出带
字段名的 `ValueError`。padding speaker ID 被忽略。

- [ ] **步骤 4：在 biggpu 官方环境运行红灯**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdt_backbone/tests/test_model.py -q
```

预期：collection FAIL，缺少 `gcnet_missing_m3_sdt_backbone.model`。

## 任务 2：实现最小对话 Transformer

**文件：**

- 创建：`gcnet_missing_m3_sdt_backbone/model.py`
- 修改：`gcnet_missing_m3_sdt_backbone/__init__.py`
- 测试：`gcnet_missing_m3_sdt_backbone/tests/test_model.py`

- [ ] **步骤 1：实现固定正弦位置编码**

```python
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0 or max_len <= 0:
            raise ValueError("dim must be positive/even and max_len positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] > self.pe.shape[0]:
            raise ValueError("sequence length exceeds max_len")
        return values + self.pe[: values.shape[0]].unsqueeze(1)
```

`pe` 是 persistent buffer，但不增加 learnable 参数。

- [ ] **步骤 2：实现 `SDTStyleConversationBackbone`**

```python
class SDTStyleConversationBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 250,
        n_speakers: int = 1,
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 5,
        ff_dim: int = 704,
        dropout: float = 0.5,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.n_speakers = int(n_speakers)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_len)
        self.speaker_embedding = nn.Embedding(
            n_speakers + 1, d_model, padding_idx=n_speakers
        )
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            PreNormTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
```

正式训练环境使用 PyTorch 1.8，因此不调用较新版本的 `norm_first` 或 `batch_first`
构造参数。先实现兼容的 Pre-LN layer：

```python
class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, values, key_padding_mask):
        normalized = self.norm1(values)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        values = values + self.attn_dropout(attended)
        normalized = self.norm2(values)
        feed_forward = self.linear2(
            self.ff_dropout(F.gelu(self.linear1(normalized)))
        )
        return values + self.output_dropout(feed_forward)
```

前向只向各层传 `src_key_padding_mask=~umask.bool()`，不传 causal mask。最终执行
`final_norm → output_projection → ReLU → valid mask`。

- [ ] **步骤 3：运行模型测试转绿**

运行任务 1 的 pytest 命令。预期：形状、padding、full-context 和错误处理全部 PASS。

- [ ] **步骤 4：提交主干原子变更**

提交只包含包入口、模型和模型测试。Commit 使用 Lore trailers，并记录 CPU 测试环境。

## 任务 3：接入 Missing-M3 并锁定参数/RNG

**文件：**

- 修改：`gcnet_missing_m3_sdt_backbone/model.py`
- 修改：`gcnet_missing_m3_sdt_backbone/tests/test_model.py`

- [ ] **步骤 1：编写集成失败测试**

使用极小维度构造 `MissingM3GraphModel` control 与 `MissingM3SDTModel`。在相同 seed
下比较以下共享 key：

```python
shared_prefixes = (
    "observed_set.",
    "teacher.",
    "missing_predictor.",
    "smax_fc.",
)
```

要求对应 tensor 精确相同。候选 state dict 中不得存在：

```text
lstm.
gru.
graph_net_temporal.
graph_net_speaker.
```

候选必须继续返回 `(logits, hidden, latents, predictions)`，且 teacher 更新次数和 target
mask 合同不变。

- [ ] **步骤 2：实现 `MissingM3SDTModel`**

```python
class MissingM3SDTModel(MissingM3GraphModel):
    def __init__(self, *args, transformer_dim=384, transformer_heads=8,
                 transformer_layers=5, transformer_ff_dim=704,
                 transformer_max_len=512, **kwargs):
        super().__init__(*args, **kwargs)
        del self.lstm
        del self.gru
        del self.graph_net_temporal
        del self.graph_net_speaker
        self.conversation_backbone = SDTStyleConversationBackbone(
            input_dim=self.latent_dim,
            output_dim=self.smax_fc.in_features,
            n_speakers=self.n_speakers,
            d_model=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=float(kwargs.get("dropout", 0.5)),
            max_len=transformer_max_len,
        )

    def encode_hidden(self, inputfeats, qmask, umask, seq_lengths,
                      pre_graph_residual=None):
        if pre_graph_residual is not None:
            raise ValueError("pre_graph_residual is unsupported")
        return self.conversation_backbone(
            inputfeats[0], qmask, umask, seq_lengths
        )
```

先执行 `super().__init__()` 的目的，是保持所有共享模块的 RNG 初始化顺序与 control
一致；旧 backbone 随后删除，不保留 dummy 参数。

- [ ] **步骤 3：编写并验证真实预算测试**

MOSI 配置下断言：

```python
registered = sum(p.numel() for p in model.conversation_backbone.parameters())
effective = registered - model.conversation_backbone.speaker_embedding.embedding_dim
assert registered == 5_869_754
assert effective == 5_869_370
assert abs(effective - 5_864_700) / 5_864_700 < 0.002
```

对 control 做一次 forward/backward，按非 `None` gradient 统计 active backbone，要求
等于 `5,864,700`。记录 dormant GRU `1,778,400` 与两个 MatchingAttention
`501,000`，不加入候选。

- [ ] **步骤 4：运行目标测试**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdt_backbone/tests/test_model.py -q
```

预期：全部 PASS。

- [ ] **步骤 5：提交 Missing-M3 集成**

Commit 记录 active 参数差异、删除的 dormant 参数和共享 RNG 等价证据。

## 任务 4：实现独立训练入口

**文件：**

- 创建：`gcnet_missing_m3_sdt_backbone/train_gcnet.py`
- 创建：`gcnet_missing_m3_sdt_backbone/tests/test_train_gcnet.py`

- [ ] **步骤 1：编写模型构建与 CLI 失败测试**

测试 `build_model()` 返回 `MissingM3SDTModel`，并锁定：

```text
dataset=CMUMOSI
hidden=100
latent_dim=256
fusion_type=slot
train_rate_mode=all
lr=5e-4
task_loss=mse
jepa_weight=0.1
readout=shared
```

CLI 只开放 seed、feature roots、output dir、device、epochs 和 skip-test；旧候选开关不
存在。`--epochs` 仅服务测试与复现，正式 runner 强制 100。

- [ ] **步骤 2：实现 `SDTTrainConfig` 与 `build_model()`**

```python
@dataclass(frozen=True)
class SDTTrainConfig(BaseTrainConfig):
    dataset: str = "CMUMOSI"
    fold: int = 1
    hidden: int = 100
    latent_dim: int = 256
    fusion_type: str = "slot"
    train_rate_mode: str = "all"
    learning_rate: float = 5e-4
    transformer_dim: int = 384
    transformer_heads: int = 8
    transformer_layers: int = 5
    transformer_ff_dim: int = 704
    transformer_max_len: int = 512
```

`__post_init__()` 拒绝任何偏离锁定 treatment 的配置。

- [ ] **步骤 3：实现独立 `run_experiment()`**

复用：

```python
from gcnet_missing_m3.train_gcnet import (
    _resolve_task_contract, _save_best_checkpoint, _schedules,
    _state_to_cpu, _write_json, evaluate_rate, get_loaders,
    mean_validation_weighted_f1, set_random_seed, train_epoch,
)
```

仅复制模型构建与 epoch orchestration，不复制 loss、mask、batch 或 metric 实现。
`metrics.json` 额外记录：

```json
{
  "backbone": "sdt-style-full-context",
  "transformer": {"d_model": 384, "heads": 8, "layers": 5, "ff_dim": 704},
  "registered_parameters": 0,
  "trainable_parameters": 0,
  "registered_backbone_parameters": 5869754,
  "active_backbone_parameters": 5869370,
  "control_active_backbone_parameters": 5864700
}
```

数值由运行时填充，示例中的 0 不得写入正式结果。

- [ ] **步骤 4：验证构建、配置和 provenance**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdt_backbone/tests/test_train_gcnet.py -q
```

预期：全部 PASS；不得启动数据集训练。

- [ ] **步骤 5：提交独立训练入口**

Commit 说明只复用现有训练 primitive，现有 control 文件无 diff。

## 任务 5：实现五种子 runner 与结果目录

**文件：**

- 创建：`gcnet_missing_m3_sdt_backbone/run_mosi.py`
- 创建：`gcnet_missing_m3_sdt_backbone/tests/test_runner.py`
- 创建：`gcnet_missing_m3_sdt_backbone/README.md`
- 创建：`gcnet_missing_m3_sdt_backbone/STATUS.md`
- 创建：`gcnet_missing_m3_sdt_backbone/results/README.md`

- [ ] **步骤 1：编写 runner 失败测试**

`build_jobs()` 必须生成且只生成 seeds 66–70。GPU assignment 为
`66→0, 67→1, 68→2, 69→0, 70→1`。每条命令必须包含：

```text
-m gcnet_missing_m3_sdt_backbone.train_gcnet
--dataset CMUMOSI
--epochs 100
--train-rate-mode all
--fusion-type slot
--lr 5e-4
```

命令中不得出现 `gcnet_missing_m3.train_gcnet`、Original 或旧 treatment 开关。

- [ ] **步骤 2：实现原子 manifest 与 resume**

每个 seed 只有在以下文件均存在且可解析时才标为完成：

```text
config.json
history.json（恰好 100 个 epoch）
metrics.json（8 个 test rate 或显式 validation-only）
train.log
```

manifest 先写 `.tmp` 再 `os.replace()`；半写结果不得继承。异常子进程有明确 return code
和 log path，runner 不无限等待。

- [ ] **步骤 3：实现并测试 dry-run**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdt_backbone/tests/test_runner.py -q
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  -m gcnet_missing_m3_sdt_backbone.run_mosi --dry-run
```

预期：pytest PASS；dry-run 打印 5 条 candidate 命令和 0 条 control 命令。

- [ ] **步骤 4：补充 README 与初始状态**

`STATUS.md` 写为 `IN PROGRESS — UNIT TESTED, FORMAL RUN NOT STARTED`。README 明确：

- 它是 SDT-style diagnostic，不是完整 SDT 复现；
- 普通 Transformer 不作为核心创新；
- 模型与结果均在本目录；
- 大 checkpoint/NPZ 不上传 GitHub。

- [ ] **步骤 5：提交 runner 与文档**

Commit 记录任务分配、resume 完整性条件和 GitHub 文件边界。

## 任务 6：一次完整验证与 CUDA profiling

**文件：**

- 修改：`gcnet_missing_m3_sdt_backbone/STATUS.md`
- 创建：`gcnet_missing_m3_sdt_backbone/results/PROFILE.json`

- [ ] **步骤 1：同步到 biggpu 的 `/data2/yb/paper` 工作树**

使用 compute-helper 的增量同步；同步后确认 remote commit 与 local commit 相同。不得在
错误的 `/data2/yb/mcv` 环境或错误工作目录运行。

- [ ] **步骤 2：运行目标与相关回归测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdt_backbone/tests -q
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  tests/test_missing_m3.py tests/test_trainer_protocol_integration.py -q
```

预期：全部 PASS。

- [ ] **步骤 3：运行唯一一次 CUDA 单 batch forward/backward**

同一个真实 MOSI batch 分别运行 Control 与 candidate，记录：

- forward wall time；
- forward + backward wall time；
- peak allocated GPU memory；
- registered/trainable/active 参数；
- finite loss 与 backbone gradient。

把结果原子写入 `results/PROFILE.json`。不运行 1-epoch 保存 smoke。

- [ ] **步骤 4：静态验证**

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m py_compile \
  gcnet_missing_m3_sdt_backbone/*.py
git diff --check
```

预期：exit code 0。

## 任务 7：运行五种子正式实验

**文件：**

- 生成：`gcnet_missing_m3_sdt_backbone/results/formal/seed_*/`
- 修改：`gcnet_missing_m3_sdt_backbone/STATUS.md`

- [ ] **步骤 1：检查 GPU 0、1、2 的真实空闲显存**

只检查一次。若 profiling 显示单任务峰值低于单卡显存的 35%，使用每卡最多 2 个并发，
一次启动 5 个 seeds；否则每卡 1 个并发，69/70 在对应 GPU 前一任务完成后自动启动。
GPU 4 不进入任务列表。

- [ ] **步骤 2：启动正式 runner**

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  -m gcnet_missing_m3_sdt_backbone.run_mosi \
  --output-root gcnet_missing_m3_sdt_backbone/results/formal \
  --gpus 0 1 2 --jobs-per-gpu 2
```

每个任务训练 100 epochs，按 validation 8-rate mean 选 checkpoint，然后测试 8 rates。
Original 不重跑。

- [ ] **步骤 3：监控有效训练进度**

每次状态更新只读取最新 epoch、validation mean 和进程状态；不重复校验 Python 环境、
NPZ 或 mask manifest。子进程退出时先检查结果完整性，再决定是否重启。

- [ ] **步骤 4：完成性审计**

要求 5 个 seed 均有 100-epoch history、8-rate test、不同 seed 的 model-init seed、相同
protocol 字段和对应 mask SHA。任何不完整 seed 只补该 seed，不重跑完整 seed。

## 任务 8：汇总、判定与 GitHub 交付

**文件：**

- 创建：`gcnet_missing_m3_sdt_backbone/results/SUMMARY.json`
- 创建：`gcnet_missing_m3_sdt_backbone/results/SUMMARY.md`
- 创建：`gcnet_missing_m3_sdt_backbone/results/PROVENANCE.json`
- 修改：`gcnet_missing_m3_sdt_backbone/STATUS.md`

- [ ] **步骤 1：逐 seed 与逐 rate 汇总**

输出 seeds 66–70 的：最佳 epoch、validation 8-rate mean、8 个 test W-F1、test mean、
high-missing mean、miss-0、prediction std 和 sign count。对照 strict validation control：

```text
Val8 78.7675
High-missing 74.9589
Miss-0 85.6461
```

- [ ] **步骤 2：执行预注册 gate**

按设计规格判定 PASS/FAIL，不根据 test 调参。失败时状态写为
`CLOSED — NO IMPROVEMENT`；通过时写为 `COMPLETE — VALIDATION GATE PASSED`。

- [ ] **步骤 3：删除 GitHub 禁止资产**

确认暂存区不含 `best.pt`、大型 NPZ、数据集、临时 manifest 或 `.tmp`。保留 JSON、
Markdown 和必要日志摘要。

- [ ] **步骤 4：最终验证**

运行目标测试、相关回归测试、`py_compile`、`git diff --check`，并读取命令输出确认成功。

- [ ] **步骤 5：Lore commit 并推送**

提交信息必须记录：唯一变量、Rejected 方案、正式 seeds/rates、验证命令、未覆盖风险。
推送：

```bash
git push github feature/m3-jepa-gcnet
```

最终报告给出 GitHub commit、独立目录、逐 seed/逐 rate 结果和下一条有证据的研究方向。
