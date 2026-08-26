# GCNet PLCI-JEPA 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变默认 Original/旧 JEPA 路径的前提下，实现 training-only Source-Anchored Pattern-Lattice JEPA，包括 pattern-balanced auxiliary view、EMA target、source anchor、一次 auxiliary GCNet forward 的 context correction 与 bounded conditional innovation。

**架构：** 新建 `gcnet_plci_jepa` 包承载 PLCI 独有模块；对现有 `GraphModel` 做最小 encode-hidden 提取，使 pattern residual 能在 pre-graph recurrent output 后注入。现有 trainer 增加显式 `--jepa-architecture plci` 路由；默认 `independent` 的参数、输出和训练语义保持不变。

**技术栈：** Python 3.8、PyTorch 1.8、PyTorch Geometric 2.0.1、NumPy、pytest。

---

## 文件职责

- 创建 `gcnet_plci_jepa/patterns.py`：六种 active pattern 编码、utterance-level balanced sampler、mask expansion。
- 创建 `gcnet_plci_jepa/modules.py`：student adapter bank、EMA teacher bank、latent normalization、bounded residual、source-anchored predictor。
- 创建 `gcnet_plci_jepa/loss.py`：pattern/target-balanced cosine JEPA loss。
- 创建 `gcnet_plci_jepa/model.py`：PLCI natural/auxiliary model API。
- 创建 `gcnet_plci_jepa/diagnostics.py`：只读 collapse/route diagnostics。
- 修改 `gcnet_modality_jepa/model.py`：提取可复用 hidden encoding，并允许 post-recurrent pattern residual；默认调用保持不变。
- 修改 `gcnet_modality_jepa/train_gcnet.py`：增加 PLCI 路由、一次 auxiliary forward、合并 loss、optimizer 后 EMA 更新及 manifest 字段。
- 创建 `tests/test_plci_patterns.py`、`tests/test_plci_modules.py`、`tests/test_plci_model.py`、`tests/test_plci_training.py`。

### 任务 1：六种 pattern 与独立 sampler

**文件：**
- 创建：`gcnet_plci_jepa/__init__.py`
- 创建：`gcnet_plci_jepa/patterns.py`
- 测试：`tests/test_plci_patterns.py`

- [ ] **步骤 1：编写失败测试**

覆盖固定映射、padding 全零、每个有效 utterance 只产生六种 active pattern、相同 generator state 可复现、改变 natural mask 不改变 auxiliary sampler。

```python
def test_balanced_sampler_uses_only_active_patterns_and_zeros_padding():
    generator = torch.Generator().manual_seed(66)
    umask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    availability = sample_balanced_patterns(umask, generator)
    assert availability.shape == (3, 2, 3)
    assert torch.equal(availability[2, 0], torch.zeros(3))
    assert all(tuple(row.tolist()) in ACTIVE_PATTERNS for row in availability[umask.T.bool()])
```

- [ ] **步骤 2：运行测试确认因模块不存在而失败**

运行：`python -m pytest -q tests/test_plci_patterns.py`

- [ ] **步骤 3：实现最少 sampler**

接口固定为：

```python
ACTIVE_PATTERNS = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
)

def sample_balanced_patterns(
    umask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return [L,B,3], sampling one active pattern per real utterance."""

def expand_modality_mask(
    availability: torch.Tensor,
    dimensions: tuple[int, int, int],
) -> torch.Tensor:
    """Expand [L,B,3] availability to concatenated feature dimensions."""
```

- [ ] **步骤 4：运行测试确认通过并提交**

运行：`python -m pytest -q tests/test_plci_patterns.py`

### 任务 2：Student adapter、EMA teacher 与 bounded residual

**文件：**
- 创建：`gcnet_plci_jepa/modules.py`
- 测试：`tests/test_plci_modules.py`

- [ ] **步骤 1：编写失败测试**

测试：adapter 只读取 observed/incomplete blocks；ATV 输出与输入精确相等；missing block 精确为零；adapter 最后一层零初始化；teacher 初始等于 student、无梯度、optimizer 后 EMA 公式精确；bounded residual norm 不超过 `kappa`。

```python
def test_bounded_residual_respects_norm_cap():
    value = torch.randn(11, 32) * 100
    bounded = bounded_residual(value, kappa=0.25)
    assert torch.all(torch.linalg.vector_norm(bounded, dim=-1) <= 0.250001)

def test_ema_update_follows_updated_student():
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    with torch.no_grad():
        next(student.parameters()).add_(1.0)
    teacher.update_from(student, tau=0.9)
    # Every teacher tensor equals 0.9 * old teacher + 0.1 * current student.
```

- [ ] **步骤 2：运行测试确认失败**

运行：`python -m pytest -q tests/test_plci_modules.py`

- [ ] **步骤 3：实现确定性 projector 与 adapter bank**

```python
class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int): ...

class StudentAdapterBank(nn.Module):
    def forward(
        self,
        full_or_masked_features: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

class EMATeacherBank(nn.Module):
    @torch.no_grad()
    def update_from(self, students: nn.ModuleDict, tau: float) -> None: ...

def normalize_latent(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: ...
def bounded_residual(value: torch.Tensor, kappa: float, eps: float = 1e-6) -> torch.Tensor: ...
```

- [ ] **步骤 4：运行测试确认通过并提交**

### 任务 3：Source-anchored predictor 与唯一 JEPA loss

**文件：**
- 修改：`gcnet_plci_jepa/modules.py`
- 创建：`gcnet_plci_jepa/loss.py`
- 测试：`tests/test_plci_predictor.py`

- [ ] **步骤 1：编写失败测试**

覆盖六种 pattern 的目标集合；单源产生两个独立 target predictions；双源产生两个 ordered paths且共享同一个 context tensor；context/innovation output 层零初始化；改变 missing target 不改变 prediction；loss 先按 utterance target 数平均。

```python
def test_audio_only_targets_text_and_visual_without_joint_target():
    outputs = predictor(student_latents, hidden, availability)
    assert set(outputs.targets_for(0, 0)) == {"text", "visual"}
    assert outputs.path_count(0, 0, "text") == 1
    assert outputs.path_count(0, 0, "visual") == 1

def test_dual_source_has_two_source_anchored_paths():
    outputs = predictor(student_latents, hidden, torch.tensor([[[1., 1., 0.]]]))
    assert outputs.path_count(0, 0, "visual") == 2
```

- [ ] **步骤 2：运行确认失败**

- [ ] **步骤 3：实现 predictor**

```python
@dataclass
class PLCITargetPrediction:
    utterance_index: int
    target_modality: int
    source_pattern: int
    anchor_modalities: tuple[int, ...]
    paths: torch.Tensor  # [one_or_two_paths, latent_dim]

@dataclass
class PLCIPredictions:
    targets: list[PLCITargetPrediction]

class SourceAnchoredPredictor(nn.Module):
    def forward(
        self,
        student_latents: dict[str, torch.Tensor],
        graph_hidden: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> PLCIPredictions: ...

def plci_jepa_loss(
    predictions: PLCIPredictions,
    teacher_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]: ...
```

Loss 只实现 `1 - dot(unit_prediction, unit_teacher)`；不得加入 path、variance、covariance 或 innovation penalty。
Loss 必须先对每个 target 的一或两条 path 平均，再对同一 utterance 的一或两个
missing targets 平均，最后对有效 utterances 平均；禁止直接对所有 paths 求全局
平均。

- [ ] **步骤 4：运行测试确认通过并提交**

### 任务 4：提取 GCNet hidden path 并接入 pattern residual

**文件：**
- 修改：`gcnet_modality_jepa/model.py`
- 创建：`gcnet_plci_jepa/model.py`
- 测试：`tests/test_plci_model.py`

- [ ] **步骤 1：锁定现有 GraphModel 回归行为**

构造同一 state dict、dropout=0 的模型，比较重构前后的 `log_prob`、`rec_outputs`、`hidden`。

```python
def test_original_forward_is_unchanged_when_pre_graph_residual_is_none():
    expected = legacy_forward_fixture(model, features, qmask, umask, lengths)
    actual = model(features, qmask, umask, lengths)
    for left, right in zip_flat_tensors(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
```

- [ ] **步骤 2：运行回归测试并保存红灯/基线证据**

- [ ] **步骤 3：提取 `encode_hidden`**

```python
def encode_hidden(
    self,
    inputfeats,
    qmask,
    umask,
    seq_lengths,
    pre_graph_residual=None,
):
    """Run Original recurrent and graph path; residual is added after RNN."""
```

默认 `pre_graph_residual=None` 时逐操作保持原顺序。

- [ ] **步骤 4：实现 PLCI model API**

```python
class PLCIJEPAGraphModel(GraphModel):
    def forward_natural(
        self, inputfeats, availability, qmask, umask, seq_lengths
    ): ...

    def forward_auxiliary(
        self, source_features, availability, qmask, umask, seq_lengths
    ): ...

    @torch.no_grad()
    def encode_teacher_targets(self, teacher_features): ...

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None: ...
```

ATV natural batch 必须 bypass adapter/pattern 并与共享参数的 Original forward 精确一致。Trainer 在调用前完成 `source_features = teacher_features * expanded_mask`；auxiliary source API 不接受完整 target。Auxiliary forward 返回 predictions 和 hidden，不计算 classifier/reconstruction head；teacher targets 由独立 no-gradient API 产生。

- [ ] **步骤 5：运行 `test_plci_model.py` 与旧 `test_modality_jepa.py` 并提交**

### 任务 5：训练循环只增加一个 auxiliary forward

**文件：**
- 修改：`gcnet_modality_jepa/train_gcnet.py`
- 测试：`tests/test_plci_training.py`

- [ ] **步骤 1：编写训练路由失败测试**

验证默认 `independent` 不构造 PLCI；`plci` 每个 train batch 调用一次 natural 和一次 auxiliary forward；validation/test 只调用 natural；总损失只含 Original cls、Original missing-only reconstruction 和 `lambda_j * plci_loss`；EMA 只在 optimizer step 后调用。

```python
def test_plci_updates_teacher_only_after_optimizer_step():
    events = []
    optimizer.step = lambda: events.append("optimizer")
    model.update_teacher = lambda tau: events.append("ema")
    run_one_plci_training_batch(...)
    assert events == ["optimizer", "ema"]
```

- [ ] **步骤 2：运行确认失败**

- [ ] **步骤 3：增加锁定 CLI**

```text
--jepa-architecture independent|plci
--plci-latent-dim 256
--plci-source-dim 256
--plci-context-rank 32
--plci-innovation-rank 32
--plci-context-cap 0.25
--plci-innovation-cap 0.25
--plci-ema-tau 0.996
--plci-aux-seed <derived when omitted>
```

默认值保持 `jepa-architecture=independent`，旧命令和 checkpoint key 不变。

`validate_training_args` 在 PLCI 模式必须要求：

```text
loss_recon=True
reccls_flag=False
lower_bound=False
reconstruction_target=missing
all_modal_recon_weight=0
stability_recon_weight=0
model_variant=addon
```

- [ ] **步骤 4：实现 loss/EMA 路由**

训练 batch：natural forward → independent auxiliary pattern sample → trainer先mask完整feature → auxiliary source forward → 独立no-gradient teacher target → 合并 backward → existing optimizer/clip step → EMA update。PLCI auxiliary 启用不依赖 natural mask rate，因此 `eta=0` 仍执行；validation/test 禁止 teacher/predictor execution。

- [ ] **步骤 5：记录 auxiliary generator state 并运行测试**

Optimizer 只接收 `requires_grad=True` 参数；重写/覆盖 train-mode 行为以保证 teacher
始终 eval。将 generator state、pattern counts、EMA step 和 PLCI config 写入 fold
metrics/run manifest；不创建正式 sweep runner，也不声明支持中途恢复训练。

- [ ] **步骤 6：提交**

### 任务 6：泄漏与训练状态诊断

**文件：**
- 创建：`gcnet_plci_jepa/diagnostics.py`
- 修改：`gcnet_modality_jepa/run_manifest.py`
- 测试：`tests/test_plci_diagnostics.py`

- [ ] **步骤 1：编写失败测试**

改变 auxiliary missing target 的完整值，使用 forward hook 同时断言 student
projector 与 GCNet 实际收到的 source tensor 不变，并断言 adapted source、hidden
和 prediction 不变而 teacher/loss 改变；记录 student/teacher std、effective
rank、Real-vs-Shuffle、context norm、innovation norm 和六 pattern counts。

- [ ] **步骤 2：实现纯诊断函数**

```python
def compute_plci_diagnostics(
    predictions,
    teacher_targets,
    student_latents,
    pattern_ids,
    shuffle_seed,
) -> dict: ...
```

诊断必须全部 detach，不加入 total loss。

- [ ] **步骤 3：运行测试并提交**

### 任务 7：一次集成验证与文档收尾

**文件：**
- 修改：`docs/superpowers/specs/2026-08-26-gcnet-plci-jepa-design.md`
- 创建：`docs/experiments/2026-08-26-plci-jepa-implementation.md`

- [ ] **步骤 1：运行唯一目标测试集合**

```bash
python -m pytest -q \
  tests/test_plci_patterns.py \
  tests/test_plci_modules.py \
  tests/test_plci_predictor.py \
  tests/test_plci_model.py \
  tests/test_plci_training.py \
  tests/test_plci_diagnostics.py \
  tests/test_modality_jepa.py \
  tests/test_full_fused_reconstruction.py
```

- [ ] **步骤 2：运行一次 CPU/官方环境前后向集成检查**

只构造合成 conversation，不读取数据集、不启动 epoch 训练。确认 finite loss/gradient、ATV bypass、一次 auxiliary forward 和 EMA 顺序。

- [ ] **步骤 3：运行静态验证**

```bash
python -m compileall -q gcnet_plci_jepa gcnet_modality_jepa
git diff --check
```

- [ ] **步骤 4：记录实现边界并提交**

文档必须明确：尚未运行正式训练、没有结果、不能进入完成版本仓库；下一步只能在用户另行要求时设计实验协议。
