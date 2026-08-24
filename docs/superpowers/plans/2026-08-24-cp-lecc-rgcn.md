# CP-LECC-RGCN 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在官方 GCNet 的 temporal 与 speaker 第一层中实现 complete-preserving low-rank edge-conditioned relational convolution，并执行预注册的 IEMOCAP-6 fold-5 十一个任务晋级门。

**架构：** PyG `RGCNConv.forward` 始终产生不变的基础输出；CP-LECC 由有序 source/target pattern、relation embedding 和双端内容交互生成四个秩 8 filter-basis 系数，只将 relation-wise mean 的动态修正加到基础输出。系数输出层零初始化，新参数初始化保存并恢复全局 RNG，因此完整设置和任意 mask 的初始前向均逐位恢复 Original。

**技术栈：** Python 3.10、PyTorch 2.2.2、PyG 2.4.0、`torch_geometric.utils.scatter`、`unittest`、NumPy、scikit-learn、SciPy、IEMOCAP-6。

---

## 文件职责

- 创建 `gcnet/missing_patterns.py`：三位 availability mask 校验、六维 contrast encoding 和 conversation-major 展平。
- 修改 `gcnet/mpfilm_rgcn.py`：从共享 pattern 工具导入，保持已归档 FiLM 行为不变。
- 创建 `gcnet/cp_lecc_rgcn.py`：只实现 `CompletePreservingLowRankECCConv`。
- 修改 `gcnet/model.py`：为 temporal/speaker 两个第一层选择 `cp_lecc`。
- 修改 `gcnet/train_gcnet.py`：把 `cp_lecc` 加入 CLI 选项。
- 修改 `experiments/mpfilm_iemocap6/run_locked_ab.py`：加入 `cp_lecc` arm，复用现有固定协议调度器。
- 创建 `experiments/cp_lecc_iemocap6/summarize_gate.py`：验证归档、计算配对结果并机械执行晋级判据。
- 创建 `tests/test_missing_patterns.py`、`tests/test_cp_lecc_rgcn.py`、`tests/test_cp_lecc_summary.py`。
- 修改 `tests/test_model_mpfilm_integration.py`、`tests/test_training_protocol.py`、`tests/test_mpfilm_runner.py`。
- 创建 `experiments/cp_lecc_iemocap6/EXPERIMENT.zh.md`：记录全部结果与判定。

## 任务 1：提取共享 missing-pattern 工具且不改变行为

**文件：**

- 创建：`gcnet/missing_patterns.py`
- 修改：`gcnet/mpfilm_rgcn.py`
- 修改：`gcnet/model.py`
- 创建：`tests/test_missing_patterns.py`
- 修改：`tests/test_mpfilm_rgcn.py`

- [ ] **步骤 1：运行现有回归测试建立基线**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_mpfilm_rgcn tests.test_model_mpfilm_integration -v
```

预期：全部 `OK`。如果失败，停止提取并先恢复 commit `01243b1` 的已验证状态。

- [ ] **步骤 2：编写共享工具导入测试并确认失败**

创建 `tests/test_missing_patterns.py`：

```python
import unittest
import torch

from missing_patterns import encode_missing_patterns, flatten_valid_node_masks


class MissingPatternUtilityTests(unittest.TestCase):
    def test_seven_patterns_and_complete_origin(self):
        masks = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=torch.float32)
        pattern, complete = encode_missing_patterns(masks)
        expected = torch.cat((torch.eye(6), torch.zeros(1, 6)), dim=0)
        torch.testing.assert_close(pattern, expected)
        torch.testing.assert_close(complete, torch.tensor([False] * 6 + [True]))

    def test_flatten_matches_conversation_major_graph_order(self):
        mask = torch.tensor([
            [[1, 0, 0], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [1, 0, 1]],
        ])
        actual = flatten_valid_node_masks(mask, [3, 1])
        expected = torch.tensor([
            [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]
        ])
        torch.testing.assert_close(actual, expected)

    def test_rejects_zero_nonbinary_and_wrong_shape(self):
        for mask in (
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[1, 2, 0]]),
            torch.tensor([[1, 0]]),
        ):
            with self.assertRaises(ValueError):
                encode_missing_patterns(mask)
```

运行该测试，预期：`ModuleNotFoundError: No module named 'missing_patterns'`。

- [ ] **步骤 3：移动而非复制共享实现**

从 `gcnet/mpfilm_rgcn.py` 原样移动 `_PATTERN_TO_COLUMN`、`_COMPLETE_PATTERN`、`_validate_node_mask`、`encode_missing_patterns` 和 `flatten_valid_node_masks` 到 `gcnet/missing_patterns.py`。导入改为：

```python
# gcnet/mpfilm_rgcn.py
from missing_patterns import encode_missing_patterns

# gcnet/model.py
from missing_patterns import flatten_valid_node_masks
from mpfilm_rgcn import MissingPatternFiLMRGCNConv
```

删除旧文件中的重复定义，禁止保留两套 pattern mapping。

- [ ] **步骤 4：验证共享工具和 FiLM 回归**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_missing_patterns tests.test_mpfilm_rgcn \
  tests.test_model_mpfilm_integration -v
```

预期：全部通过；现有 MPFiLM complete parity 与 edgewise activation 测试不变。

- [ ] **步骤 5：提交公共语义提取**

```bash
git add gcnet/missing_patterns.py gcnet/mpfilm_rgcn.py gcnet/model.py \
  tests/test_missing_patterns.py tests/test_mpfilm_rgcn.py
git commit -m "Share missing-pattern semantics across graph candidates"
```

## 任务 2：以 TDD 实现 CP-LECC 核心卷积

**文件：**

- 创建：`gcnet/cp_lecc_rgcn.py`
- 创建：`tests/test_cp_lecc_rgcn.py`

- [ ] **步骤 1：编写 complete parity、参数量和 RNG 的失败测试**

创建 `tests/test_cp_lecc_rgcn.py`：

```python
import unittest
import torch
from torch import nn
from torch_geometric.nn import RGCNConv

from cp_lecc_rgcn import CompletePreservingLowRankECCConv


def copy_base(source, target):
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.root.copy_(source.root)
        target.bias.copy_(source.bias)


class CPLECCCoreTests(unittest.TestCase):
    def test_complete_forward_backward_is_bitwise_pyg(self):
        torch.manual_seed(11)
        x_ref = torch.randn(4, 5, requires_grad=True)
        x_new = x_ref.detach().clone().requires_grad_(True)
        edges = torch.tensor([[0, 2, 1, 3], [1, 1, 2, 2]])
        types = torch.tensor([0, 0, 1, 1])
        mask = torch.ones(4, 3)
        reference = RGCNConv(5, 3, 2)
        candidate = CompletePreservingLowRankECCConv(
            5, 3, 2, content_dim=4, relation_dim=3,
            generator_hidden=5, num_bases=2, basis_rank=2,
        )
        copy_base(reference, candidate)
        expected = reference(x_ref, edges, types)
        actual = candidate(x_new, edges, types, mask)
        self.assertTrue(torch.equal(actual, expected))
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        for left, right in (
            (x_ref.grad, x_new.grad),
            (reference.weight.grad, candidate.weight.grad),
            (reference.root.grad, candidate.root.grad),
            (reference.bias.grad, candidate.bias.grad),
        ):
            self.assertTrue(torch.equal(left, right))

    def test_constructor_does_not_advance_global_rng(self):
        torch.manual_seed(29)
        reference = RGCNConv(5, 3, 2)
        after_reference = nn.Linear(7, 4)
        torch.manual_seed(29)
        candidate = CompletePreservingLowRankECCConv(5, 3, 2)
        after_candidate = nn.Linear(7, 4)
        self.assertTrue(torch.equal(candidate.weight, reference.weight))
        self.assertTrue(torch.equal(after_candidate.weight, after_reference.weight))
        self.assertTrue(torch.equal(after_candidate.bias, after_reference.bias))

    def test_locked_parameter_budget(self):
        temporal = CompletePreservingLowRankECCConv(400, 100, 3)
        speaker = CompletePreservingLowRankECCConv(400, 100, 4)
        base_t = RGCNConv(400, 100, 3)
        base_s = RGCNConv(400, 100, 4)
        extra = sum(p.numel() for p in temporal.parameters())
        extra += sum(p.numel() for p in speaker.parameters())
        extra -= sum(p.numel() for p in base_t.parameters())
        extra -= sum(p.numel() for p in base_s.parameters())
        self.assertEqual(extra, 60_672)
```

运行：

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_cp_lecc_rgcn -v
```

预期：因 `cp_lecc_rgcn` 不存在失败。

- [ ] **步骤 2：实现构造函数、RNG 恢复和 complete 快速路径**

公共类固定为 `CompletePreservingLowRankECCConv(RGCNConv)`。构造签名为 `__init__(self, in_channels: int, out_channels: int, num_relations: int, content_dim: int = 16, relation_dim: int = 8, generator_hidden: int = 32, num_bases: int = 4, basis_rank: int = 8) -> None`；前向签名为 `forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor, node_mask: Tensor) -> Tensor`。步骤 2 定义全部参数和初始化，步骤 5 定义完整前向体。

参数形状必须是：

```python
self.target_content = nn.Parameter(torch.empty(in_channels, content_dim))
self.source_content = nn.Parameter(torch.empty(in_channels, content_dim))
self.relation_embedding = nn.Parameter(torch.empty(num_relations, relation_dim))
self.generator_hidden_weight = nn.Parameter(
    torch.empty(18 + relation_dim + content_dim, generator_hidden)
)
self.generator_hidden_bias = nn.Parameter(torch.empty(generator_hidden))
self.generator_output_weight = nn.Parameter(torch.empty(generator_hidden, num_bases))
self.generator_output_bias = nn.Parameter(torch.empty(num_bases))
self.basis_left = nn.Parameter(torch.empty(num_bases, in_channels, basis_rank))
self.basis_right = nn.Parameter(torch.empty(num_bases, basis_rank, out_channels))
```

新增参数初始化必须保存并恢复 `torch.get_rng_state()`；矩阵类非输出参数 Glorot，隐藏 bias 与输出 weight/bias 为零。全完整时直接：

```python
return super().forward(x, edge_index, edge_type)
```

- [ ] **步骤 3：运行步骤 1 测试并确认第一组转绿**

预期：complete parity、RNG 和参数量通过。

- [ ] **步骤 4：编写动态行为和非死亡梯度失败测试**

加入：

```python
def test_zero_residual_starts_at_pyg_but_output_layer_learns(self):
    torch.manual_seed(31)
    layer = CompletePreservingLowRankECCConv(3, 2, 1)
    reference = RGCNConv(3, 2, 1)
    copy_base(layer, reference)
    x = torch.randn(3, 3, requires_grad=True)
    edges = torch.tensor([[0, 1], [2, 2]])
    types = torch.zeros(2, dtype=torch.long)
    mask = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    actual = layer(x, edges, types, mask)
    expected = reference(x, edges, types)
    self.assertTrue(torch.equal(actual, expected))
    actual.sum().backward()
    self.assertGreater(layer.generator_output_weight.grad.abs().sum().item(), 0)
    self.assertTrue(torch.isfinite(layer.generator_output_weight.grad).all())
```

再用固定手工参数分别验证：`A->T != T->A`、改变 source 改变修正、改变 target 改变修正、relation 0 与 1 不同、mixed graph 中 ATV-to-ATV 修正为零。

- [ ] **步骤 5：实现 descriptor、basis message 和 correction aggregation**

实现主体必须是：

```python
base_output = super().forward(x, edge_index, edge_type)
pattern, complete = encode_missing_patterns(node_mask.to(x))
source, target = edge_index
content_pair = (
    (x[target] @ self.target_content)
    * (x[source] @ self.source_content)
)
descriptor = torch.cat((
    pattern[target], pattern[source],
    pattern[target] * pattern[source],
    self.relation_embedding[edge_type], content_pair,
), dim=-1)
hidden = torch.relu(
    descriptor @ self.generator_hidden_weight + self.generator_hidden_bias
)
coefficients = torch.tanh(
    hidden @ self.generator_output_weight + self.generator_output_bias
)
left = torch.einsum("ed,kdr->ekr", x[source], self.basis_left)
basis_messages = torch.einsum("ekr,kro->eko", left, self.basis_right)
active = (~(complete[source] & complete[target])).to(x.dtype)
edge_correction = (
    coefficients.unsqueeze(-1) * basis_messages
).sum(dim=1) * active.unsqueeze(-1)
```

对每个 relation 独立：

```python
correction = x.new_zeros((x.size(0), self.out_channels))
for relation_id in range(self.num_relations):
    selected = edge_type == relation_id
    if bool(selected.any()):
        correction = correction + scatter(
            edge_correction[selected], target[selected], dim=0,
            dim_size=x.size(0), reduce="mean",
        )
return base_output + correction
```

禁止重新计算基础 `x[source] @ weight[r]`。

- [ ] **步骤 6：补齐固定值聚合、同质 pattern、校验和 CUDA 测试**

明确覆盖同 relation mean、不同 relation 分别 mean 后求和、root/bias、A-only 同质邻域、非法 mask/edge/relation，以及 CUDA FP32 有限 forward/backward。人工断言不得依赖随机“不相等”。

- [ ] **步骤 7：运行核心与旧回归测试**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_cp_lecc_rgcn tests.test_missing_patterns \
  tests.test_mpfilm_rgcn -v
git diff --check
```

预期：全部通过，diff check 无输出。

- [ ] **步骤 8：提交核心卷积**

```bash
git add gcnet/cp_lecc_rgcn.py tests/test_cp_lecc_rgcn.py
git commit -m "Distinguish GCNet sources with edge-conditioned filter bases"
```

## 任务 3：集成 GCNet 双分支且保持 Original 初始化

**文件：**

- 修改：`gcnet/model.py`
- 修改：`gcnet/train_gcnet.py`
- 修改：`tests/test_model_mpfilm_integration.py`
- 修改：`tests/test_training_protocol.py`

- [ ] **步骤 1：编写双分支与总参数失败测试**

在 `tests/test_model_mpfilm_integration.py` 增加：

```python
def test_cp_lecc_replaces_both_first_layers_only(self):
    model = self._model("cp_lecc")
    self.assertIsInstance(
        model.graph_net_temporal.conv1,
        CompletePreservingLowRankECCConv,
    )
    self.assertIsInstance(
        model.graph_net_speaker.conv1,
        CompletePreservingLowRankECCConv,
    )
    self.assertEqual(model.graph_net_temporal.conv1.num_relations, 3)
    self.assertEqual(model.graph_net_speaker.conv1.num_relations, 4)
```

用正式归档中的真实 feature dimensions 构造 `hidden=200` 模型并断言：

```python
self.assertEqual(sum(p.numel() for p in model.parameters()), 34_200_838)
```

运行聚焦测试，预期因未知 variant 或缺少 import 失败。

- [ ] **步骤 2：实现最小 variant wiring**

在 `gcnet/model.py`：

```python
from cp_lecc_rgcn import CompletePreservingLowRankECCConv

if graph_conv_variant == "original":
    self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
elif graph_conv_variant == "cp_lecc":
    self.conv1 = CompletePreservingLowRankECCConv(
        num_features, hidden_size, num_relations
    )
else:
    self.conv1 = MissingPatternFiLMRGCNConv(
        num_features, hidden_size, num_relations,
        variant=graph_conv_variant,
    )
```

forward 中 `cp_lecc` 与 MPFiLM 一样接收 `node_mask`。在 `gcnet/train_gcnet.py` 的 choices 仅追加 `'cp_lecc'`，禁止改变第二层或后续结构。

- [ ] **步骤 3：增加全模型 RNG/complete 等价测试**

分别在构造 `original` 和 `cp_lecc` 前重设同一 seed，验证全部同名、同 shape 的非 CP-LECC 参数逐位相同。全 ATV 输入下，先把 CP-LECC 新参数人工设为非零，再验证 log probabilities、reconstruction 和 hidden 与 Original 逐位相同。

- [ ] **步骤 4：运行集成与训练协议测试**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_model_mpfilm_integration \
  tests.test_training_protocol tests.test_cp_lecc_rgcn -v
```

预期：全部通过。

- [ ] **步骤 5：提交双分支集成**

```bash
git add gcnet/model.py gcnet/train_gcnet.py \
  tests/test_model_mpfilm_integration.py tests/test_training_protocol.py
git commit -m "Apply edge-conditioned filtering to both GCNet relation graphs"
```

## 任务 4：锁定 11-job 调度与机械判定

**文件：**

- 修改：`experiments/mpfilm_iemocap6/run_locked_ab.py`
- 修改：`tests/test_mpfilm_runner.py`
- 创建：`experiments/cp_lecc_iemocap6/__init__.py`
- 创建：`experiments/cp_lecc_iemocap6/summarize_gate.py`
- 创建：`tests/test_cp_lecc_summary.py`

- [ ] **步骤 1：先写 arm 与网格失败测试**

在 runner 测试增加：

```python
def test_cp_lecc_arm_maps_to_training_variant(self):
    job = build_jobs(
        "formal", Path("/tmp/results"), arms=("cp_lecc",),
        rates=(0.5,), seeds=(66,),
    )[0]
    command = build_command(
        job, Path("/env/python"), Path("/repo"),
        Path("/data"), Path("/banks"),
    )
    index = command.index("--graph-conv-variant") + 1
    self.assertEqual(command[index], "cp_lecc")

def test_cp_lecc_gate_has_exactly_eleven_jobs(self):
    complete = build_jobs(
        "formal", Path("/tmp/results"), arms=("cp_lecc",),
        rates=(0.0,), seeds=(66,),
    )
    missing = build_jobs(
        "formal", Path("/tmp/results"), arms=("cp_lecc",),
        rates=(0.5, 0.7), seeds=(66, 67, 68, 69, 70),
    )
    self.assertEqual(len(complete + missing), 11)
```

预期：arm validation 失败。

- [ ] **步骤 2：只增加 runner 映射**

```python
ARM_TO_GRAPH_VARIANT["cp_lecc"] = "cp_lecc"
```

不改变 runner 默认 Original/Full grid，也不增加新调度逻辑。

- [ ] **步骤 3：定义 summary 纯函数并先写合成测试**

`summarize_gate.py` 必须导出三个固定接口：`archive_metrics(path: Path) -> dict`、`assert_complete_archive_equal(candidate: Path, original: Path) -> None` 和 `paired_gate(candidate_rows: list[dict], original_rows: list[dict], full_rows: list[dict]) -> dict`。

`tests/test_cp_lecc_summary.py` 必须合成 5 seeds × 2 rates：

```python
def test_gate_passes_only_when_all_registered_conditions_hold(self):
    self.assertTrue(paired_gate(candidate, original, full)["promote"])

def test_each_single_failure_rejects_promotion(self):
    # 分别测试：某 rate 负、少于四胜、gain < 0.005、
    # 不胜 Full、coverage < 6、mask hash 不同。
    for mutation in mutations:
        with self.subTest(mutation=mutation.name):
            self.assertFalse(
                paired_gate(mutation.rows, original, full)["promote"]
            )
```

`archive_metrics` 固定使用 `folder_savewhole[0][-1]`，拼接 `test_labels` 和 `test_preds`，weighted F1 使用 `f1_score(labels, predictions, average="weighted")`。

- [ ] **步骤 4：实现 complete 深比较和 gate 判定**

complete 比较字段固定为：

```text
best_epoch_index
folder_losswhole
test_labels
test_preds
test_hiddens
test_fmask
mask_bank_manifest.sha256
```

`paired_gate` 先按 `(rate, seed)` 对齐，再在每个 seed 内平均 0.5/0.7。输出必须包含逐任务 metrics、逐 seed delta、rate means、wins、coverage、dominant ratio 和每条 gate condition 的布尔值，禁止只输出最终 `promote`。

纯函数中的 `promote` 必须严格等于以下条件的逻辑与：0.5 相对 Original 的 mean delta `>= 0`；0.7 相对 Original 的 mean delta `>= 0`；五个 seed-level 两率平均 delta 的均值 `>= 0.005`；至少 4/5 seeds 的 delta 为正；候选 seed-level 均值严格大于 Full；全部 candidate rows 的 `coverage == 6`；全部配对 mask hash 相同。不得使用四舍五入后的显示值判断。

- [ ] **步骤 5：运行 runner/summary 测试并提交**

```bash
PYTHONPATH=.:gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_mpfilm_runner tests.test_cp_lecc_summary -v
```

预期：全部通过。

```bash
git add experiments/mpfilm_iemocap6/run_locked_ab.py \
  experiments/cp_lecc_iemocap6 tests/test_mpfilm_runner.py \
  tests/test_cp_lecc_summary.py
git commit -m "Pre-register the CP-LECC promotion decision"
```

## 任务 5：完成实现验证并锁定实验 commit

**文件：**

- 检查：全部上述实现和测试文件
- 不创建新的模型变体

- [ ] **步骤 1：运行全套单元/集成测试**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest discover -s tests -v
```

预期：0 failure、0 error；当前机器有 CUDA，CUDA 测试不得被意外 skip。

- [ ] **步骤 2：运行静态和工作树检查**

```bash
git diff --check
git status --short
git log -5 --oneline
```

预期：diff check 无输出；所有实现/测试变更已提交；记录实验 HEAD。

- [ ] **步骤 3：运行单任务 complete audit**

```bash
PYTHONPATH=.:gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -u \
  experiments/mpfilm_iemocap6/run_locked_ab.py \
  --stage formal \
  --output-root /data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1 \
  --data-root /data2/yb/paper/GCNet_TPAMI/dataset/IEMOCAP \
  --mask-bank-root /data2/yb/paper/experiments/mpfilm_iemocap6_20260824/mask_banks \
  --gpus 0 --workers-per-gpu 1 --arms cp_lecc --rates 0.0 --seeds 66
```

使用 `assert_complete_archive_equal` 与以下归档比较：

```text
/data2/yb/paper/experiments/mpfilm_iemocap6_20260824/formal_v1/formal/original/miss_0p0/seed_66/fold_5/saved/*.npz
```

预期：best epoch、100-epoch loss、labels、logits、hidden、feature mask 和 mask hash 全部逐元素相同。失败则停止，不启动 missing jobs。

- [ ] **步骤 4：记录实验锁定元数据**

在 summary JSON 写入 command、git HEAD、Python/PyTorch/PyG/CUDA 版本、GPU 型号和 mask hash。现有 runner 保存 command/status；禁止在此步修改模型。

## 任务 6：运行十个 missing jobs 并执行停止门

**文件：**

- 生成：`/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1/formal/cp_lecc/`
- 创建：`experiments/cp_lecc_iemocap6/EXPERIMENT.zh.md`

- [ ] **步骤 1：启动唯一 missing grid**

```bash
PYTHONPATH=.:gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -u \
  experiments/mpfilm_iemocap6/run_locked_ab.py \
  --stage formal \
  --output-root /data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1 \
  --data-root /data2/yb/paper/GCNet_TPAMI/dataset/IEMOCAP \
  --mask-bank-root /data2/yb/paper/experiments/mpfilm_iemocap6_20260824/mask_banks \
  --gpus 0 1 2 3 --workers-per-gpu 3 \
  --arms cp_lecc --rates 0.5 0.7 --seeds 66 67 68 69 70
```

预期：10/10 status 为 success、10 个 NPZ；每卡同时不超过 3 个进程。训练期间不修改代码、mask、超参数或判据。

- [ ] **步骤 2：运行机械汇总**

```bash
PYTHONPATH=.:gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  experiments/cp_lecc_iemocap6/summarize_gate.py \
  --candidate-root /data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1/formal/cp_lecc \
  --original-root /data2/yb/paper/experiments/mpfilm_iemocap6_20260824/formal_v1/formal/original \
  --full-root /data2/yb/paper/experiments/mpfilm_iemocap6_20260824/formal_v1/formal/full \
  --complete-candidate /data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1/formal/cp_lecc/miss_0p0/seed_66/fold_5/saved \
  --output-json /data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/gate_v1/gate_summary.json
```

预期：程序退出 0 并打印 `PROMOTE` 或 `REJECT`；JSON 包含全部判据证据。`REJECT` 是有效实验结果，不是程序失败。

- [ ] **步骤 3：写实验过程 Markdown**

`EXPERIMENT.zh.md` 必须包含：

- 研究假设和 ECC 迁移边界；
- 实现 commit 与精确参数量；
- 11/11 任务状态和计算资源；
- complete archive 逐元素审计；
- 每个 rate/seed 的 F1、accuracy、coverage、dominant ratio、best epoch；
- candidate−Original、candidate−Full 的均值±标准差；
- seed-level 两率平均差和胜/平/负；
- 每条预注册条件的真值；
- 明确写 `PROMOTE` 或 `REJECT`；
- 官方 val=test 限制和 HGDN 未决撞车风险。

- [ ] **步骤 4：按判定停止或交接**

若 `REJECT`：不创建 full-grid jobs，不调整架构，提交负结果文档。

若 `PROMOTE`：只创建下一阶段计划；本计划不直接启动另外 30 个 rate/seed，也不实现消融。

- [ ] **步骤 5：最终验证与提交**

```bash
PYTHONPATH=gcnet /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest discover -s tests -v
git diff --check
git status --short
```

提交：

```bash
git add experiments/cp_lecc_iemocap6/EXPERIMENT.zh.md
git commit -m "Record whether edge-conditioned filtering clears the GCNet gate"
```

## 计划自检

- 规格中的 source pattern、target pattern、pattern interaction、relation 和双端内容均由任务 2 的 descriptor 与行为测试覆盖。
- complete-preserving 由模块 parity、全模型 RNG parity 和正式归档逐元素比较三重覆盖。
- 参数预算、非死亡梯度、relation-wise mean、root/bias、节点顺序和设备均有明确测试。
- 第一阶段严格为 11 个任务；runner 默认网格不被改写，完整实验不会提前启动。
- 晋级判据由纯函数和合成反例锁定，结果出来后不能人工修改。
- 计划不实现 pattern-only、parameter-matched、attention、动态图、额外损失或 JEPA。
- 失败是合法终态；失败后没有“调一个超参数再试”的隐含步骤。
