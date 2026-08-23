# GCNet Full-Fused Reconstruction 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在参数、初始化、mask 和分类路径完全一致的条件下，对比 IEMOCAPSix GCNet 的 missing-only reconstruction 与 full-fused reconstruction，并完成 fold5、8 个 missing rate、10 seeds 的 160-job 正式实验。

**架构：** 两个条件实例化完全相同的 `GraphModel` 和 `linear_rec`。Baseline 仅在每个 utterance 实际缺失的模态上计算 MSE；FFR 在至少缺一个模态的 utterance 上，对完整 Audio/Text/Visual concat 的三个模态分别按维度归一化 MSE 后等权平均。重构结果不进入分类器，实验只改变 loss selection。

**技术栈：** Python 3.10、PyTorch 2.2、pytest、现有 GCNet official fold5 协议、biggpu 多卡调度。

---

## 文件边界

- 修改 `gcnet_modality_jepa/loss.py`：新增单一职责的 `FullFusedReconLoss`。
- 修改 `gcnet_modality_jepa/train_gcnet.py`：增加 loss mode CLI、验证、训练路由和结果字段。
- 修改 `gcnet_modality_jepa/run_manifest.py`：若 schema 对 method 字段有限制，登记 reconstruction target，保持旧 manifest 可读。
- 创建 `tests/test_full_fused_reconstruction.py`：loss、padding、零缺失、参数和梯度 parity。
- 创建 `tests/test_iemocap6_full_fused_runner.py`：160-job 矩阵、GPU4 拒绝、配对命令与完成证据测试。
- 创建 `scripts/run_iemocap6_full_fused_sweep.py`：隔离的多卡 paired runner。
- 创建 `docs/experiments/2026-08-23-iemocap6-full-fused-reconstruction.md`：smoke、正式命令、结果和统计结论。

### 任务 1：为 full-fused loss 建立失败测试

**文件：**
- 创建：`tests/test_full_fused_reconstruction.py`
- 测试：`gcnet_modality_jepa/loss.py`

- [ ] **步骤 1：编写 full-fused selection 和归一化测试**

```python
def test_full_fused_loss_reconstructs_all_modalities_when_any_is_missing():
    predicted = [torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])]
    target = [torch.zeros_like(predicted[0])]
    availability = [torch.tensor([[[0.0, 1.0, 1.0]]])]
    umask = torch.ones(1, 1)
    loss = FullFusedReconLoss()(
        predicted, target, availability, umask, adim=1, tdim=1, vdim=2
    )
    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)
```

- [ ] **步骤 2：编写 padding、fully-observed 和 empty-selection 测试**

```python
def test_full_fused_loss_ignores_padding_and_fully_observed_utterances():
    predicted = torch.full((3, 1, 4), 100.0, requires_grad=True)
    target = torch.zeros_like(predicted)
    predicted.data[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    availability = torch.tensor([
        [[0.0, 1.0, 1.0]],  # selected real utterance
        [[1.0, 1.0, 1.0]],  # fully observed real utterance
        [[0.0, 0.0, 0.0]],  # padded utterance
    ])
    umask = torch.tensor([[1.0, 1.0, 0.0]])
    loss = FullFusedReconLoss()(
        [predicted], [target], [availability], umask, 1, 1, 2
    )
    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)

def test_full_fused_loss_is_differentiable_zero_without_missing_targets():
    predicted = torch.randn(2, 1, 6, requires_grad=True)
    loss = FullFusedReconLoss()(
        [predicted], [torch.randn_like(predicted)],
        [torch.ones(2, 1, 3)], torch.ones(1, 2), 2, 2, 2,
    )
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(predicted.grad) == 0
```

- [ ] **步骤 3：运行测试并确认因类不存在而失败**

运行：

```bash
pytest -q tests/test_full_fused_reconstruction.py
```

预期：collection/import FAIL，指出 `FullFusedReconLoss` 不存在。

### 任务 2：实现最小 full-fused loss

**文件：**
- 修改：`gcnet_modality_jepa/loss.py`
- 测试：`tests/test_full_fused_reconstruction.py`

- [ ] **步骤 1：实现 modality-balanced complete-target MSE**

```python
class FullFusedReconLoss(nn.Module):
    def forward(self, reconstruction, target, input_mask, umask,
                adim, tdim, vdim):
        predicted = reconstruction[0]
        expected = target[0].detach()
        availability = input_mask[0]
        real = umask.transpose(0, 1).bool()
        selected = real & (availability < 1).any(dim=-1)
        if not selected.any():
            return predicted.sum() * 0.0
        losses = []
        for pred_m, target_m in zip(
            torch.split(predicted, (adim, tdim, vdim), dim=-1),
            torch.split(expected, (adim, tdim, vdim), dim=-1),
        ):
            losses.append((pred_m[selected] - target_m[selected]).square().mean())
        return torch.stack(losses).mean()
```

- [ ] **步骤 2：运行新 loss 测试**

运行：`pytest -q tests/test_full_fused_reconstruction.py`

预期：全部 PASS。

- [ ] **步骤 3：运行现有 reconstruction 回归测试**

运行：

```bash
pytest -q tests/test_reconstruction_normalization.py tests/test_common_stability_path.py
```

预期：全部 PASS，旧 `MaskedReconLoss` 行为不变。

- [ ] **步骤 4：提交 loss 单元**

只提交 `loss.py` 和新测试，commit 记录 full-fused loss 的选择与归一化语义。

### 任务 3：训练路由、参数 parity 与 missing=0 parity

**文件：**
- 修改：`gcnet_modality_jepa/train_gcnet.py`
- 修改：`gcnet_modality_jepa/run_manifest.py`
- 修改：`tests/test_full_fused_reconstruction.py`
- 修改或创建：`tests/test_trainer_protocol_integration.py`

- [ ] **步骤 1：先写失败的 CLI 与参数 parity 测试**

```python
def test_reconstruction_target_cli_defaults_to_missing_only():
    args = build_argument_parser().parse_args([])
    assert args.reconstruction_target == "missing"

def test_full_fused_mode_does_not_change_model_parameters():
    missing_model = build_model(make_args(reconstruction_target="missing"), 4, 5, 6)
    fused_model = build_model(make_args(reconstruction_target="full_fused"), 4, 5, 6)
    assert list(missing_model.state_dict()) == list(fused_model.state_dict())
    assert [p.shape for p in missing_model.parameters()] == [p.shape for p in fused_model.parameters()]
    assert sum(p.numel() for p in missing_model.parameters()) == sum(p.numel() for p in fused_model.parameters())
```

- [ ] **步骤 2：运行并确认 CLI 测试因参数不存在而失败**

运行：`pytest -q tests/test_full_fused_reconstruction.py`

预期：FAIL，`Namespace` 无 `reconstruction_target`。

- [ ] **步骤 3：增加 loss mode CLI 和训练路由**

```python
parser.add_argument(
    "--reconstruction-target",
    choices=("missing", "full_fused"),
    default="missing",
)

reconstruction_loss = (
    full_fused_rec_loss
    if args.reconstruction_target == "full_fused"
    else rec_loss
)
loss2 = reconstruction_loss(
    recon_input_features, input_features, input_features_mask,
    umask, adim, tdim, vdim,
)
```

实例化和 CUDA 路由必须同时包含 `FullFusedReconLoss`，但不得改变 model construction。

- [ ] **步骤 4：记录结果和 manifest 字段**

`fold_metrics.json` 增加：

```json
{"reconstruction_target": "missing|full_fused"}
```

run manifest 的 method section 同样记录该字段；旧 manifest 缺字段时仍按 `missing` 解释。

- [ ] **步骤 5：增加 missing=0 loss/logit/gradient parity 测试**

测试使用同一 state dict、同一 batch 和全 1 availability mask，验证：

```python
assert missing_loss.item() == fused_loss.item() == 0.0
assert torch.equal(logits_missing, logits_fused)
assert_all_parameter_gradients_equal(model_missing, model_fused, atol=1e-7)
```

- [ ] **步骤 6：运行定向和全量测试**

运行：

```bash
pytest -q tests/test_full_fused_reconstruction.py tests/test_trainer_protocol_integration.py tests/test_run_manifest.py tests/test_run_manifest_integration.py
pytest -q
```

预期：定向测试和全套测试全部 PASS。

- [ ] **步骤 7：提交训练路由**

提交仅包含 CLI、loss routing、manifest 和相应测试。

### 任务 4：实现可审计的 160-job paired runner

**文件：**
- 创建：`scripts/run_iemocap6_full_fused_sweep.py`
- 创建：`tests/test_iemocap6_full_fused_runner.py`

- [ ] **步骤 1：先写失败的矩阵与命令测试**

```python
def test_formal_matrix_has_160_paired_jobs(tmp_path):
    jobs = build_jobs(tmp_path, python="python", gpus=(0, 1), jobs_per_gpu=3)
    assert len(jobs) == 160
    assert {job.condition for job in jobs} == {"baseline", "full_fused"}
    assert len({(job.rate, job.seed) for job in jobs}) == 80

def test_pair_differs_only_in_reconstruction_target_and_output(tmp_path):
    baseline, fused = paired_jobs(tmp_path, rate=0.4, seed=66)
    assert normalized_command_diff(baseline.command, fused.command) == {
        "--reconstruction-target": ("missing", "full_fused"),
        "--output-dir": (str(baseline.output_dir), str(fused.output_dir)),
    }

def test_runner_rejects_gpu4(tmp_path):
    with pytest.raises(ValueError, match="GPU 4"):
        build_jobs(tmp_path, python="python", gpus=(4,))
```

- [ ] **步骤 2：运行并确认 runner import 失败**

运行：`pytest -q tests/test_iemocap6_full_fused_runner.py`

预期：FAIL，runner 模块不存在。

- [ ] **步骤 3：实现矩阵和 paired shared initialization**

runner 固定：

```python
DATASET = "IEMOCAPSix"
RATES = tuple(index / 10 for index in range(8))
SEEDS = tuple(range(66, 76))
CONDITIONS = ("baseline", "full_fused")
```

每个 `(rate, seed)` pair 创建一次 shared initialization checkpoint，并在
两条命令中传入相同的 `--shared-init-checkpoint` 和
`--require-shared-init-hash`。两条命令复用相同 mask seed 和 official fold5。

- [ ] **步骤 4：实现隔离、恢复和完成验证**

runner 必须提供：

- output-root 独占 claim；
- 单 GPU 最多三个并发任务；
- GPU4 硬拒绝；
- `--rates`、`--seeds`、`--epochs` 用于 smoke；
- status 原子写入；
- 完成条件同时要求 returncode 0、fold metrics、manifest、参数 parity、
  shared-init hash 和 mask hashes；
- 已完成任务可安全 resume，不覆盖证据。

- [ ] **步骤 5：运行 runner 测试与全量测试**

运行：

```bash
pytest -q tests/test_iemocap6_full_fused_runner.py
pytest -q
```

预期：全部 PASS。

- [ ] **步骤 6：提交 runner**

提交 runner 和 runner 测试，不修改旧 640-run scheduler。

### 任务 5：远程 smoke 与正式多卡实验

**远程目录：**
- 代码：`/data2/yb/paper/GCNet_iemocap6_full_fused_20260823`
- Smoke：`/data2/yb/experiments/gcnet_iemocap6_full_fused_smoke_20260823`
- Formal：`/data2/yb/experiments/gcnet_iemocap6_full_fused_10seed_20260823`

- [ ] **步骤 1：在官方环境运行全部测试**

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest -q
```

预期：全套 PASS，无 collection error。

- [ ] **步骤 2：检查 GPU 所有权和空闲状态**

仅使用空闲且没有其他用户进程的 GPU；永不使用 GPU4。每张卡最多并发三个 job。

- [ ] **步骤 3：运行一轮 paired smoke**

```bash
python -u scripts/run_iemocap6_full_fused_sweep.py \
  --output-root /data2/yb/experiments/gcnet_iemocap6_full_fused_smoke_20260823 \
  --gpus 5 --jobs-per-gpu 2 --rates 0.0,0.4 --seeds 66 \
  --epochs 1 --python /data2/yb/reproduction_envs/gcnet-official/bin/python
```

预期：4/4 jobs、2/2 pair audits 完成；missing=0 参数、logit、loss 和 mask parity 通过。

- [ ] **步骤 4：启动 160-job 正式实验**

```bash
python -u scripts/run_iemocap6_full_fused_sweep.py \
  --output-root /data2/yb/experiments/gcnet_iemocap6_full_fused_10seed_20260823 \
  --gpus 0,1,2,3,5,6,7 --jobs-per-gpu 3 --epochs 100 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python
```

启动前按实际占用移除非空闲 GPU。runner 运行期间持续检查完成数、错误数、显存和配对审计；不终止其他用户进程。

- [ ] **步骤 5：验证 160/160 完整性**

要求：

```text
complete_jobs = 160
worker_errors = []
paired_audits = 80
paired_audit_failures = 0
```

### 任务 6：统计分析与实验记录

**文件：**
- 创建：`docs/experiments/2026-08-23-iemocap6-full-fused-reconstruction.md`

- [ ] **步骤 1：汇总每个 missing rate 的十种子结果**

记录 Baseline/FFR Weighted-F1 mean、sample SD、paired delta、10-seed win
count、每个 seed 的原始分数。

- [ ] **步骤 2：执行配对统计与 collapse audit**

每个 missing rate 使用 paired Wilcoxon，八次检验做 Holm correction；种子
坍塌沿用正式 640-run 报告的 median/MAD 规则，主表不得删除 seed。

- [ ] **步骤 3：写入可追溯报告**

报告必须包含 source commit、Python/Torch/CUDA、命令、参数量、运行目录、
完整性证据、结果表、统计结论和剩余风险。

- [ ] **步骤 4：最终验证**

运行 `git diff --check`、全量 pytest，并核对报告中的每个数值可追溯到
对应 `fold_metrics.json`。
