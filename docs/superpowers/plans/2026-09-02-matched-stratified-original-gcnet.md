# Matched Stratified Original GCNet 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 新增一个独立的 Original GCNet 正式控制入口，在与 Missing-M3 完全相同的 conversation-level stratified 训练和固定八率评估协议下，运行官方 GCNet 架构及其 masked reconstruction objective。

**架构：** 独立 `gcnet_original_stratified` 包只包含参数为零新增的 Original GraphModel 适配器和专用训练器；mask、数据加载、指标、固定评估 schedule 与 checkpoint 工具复用已验证的 Missing-M3 协议代码。训练每个 source conversation 只有一个 masked view，损失固定为 task loss 加 corrected formal masked reconstruction loss。

**技术栈：** Python 3、PyTorch、PyTorch Geometric、NumPy、scikit-learn、pytest、远程 V100、现有 deterministic mask schedule。

---

## 文件职责

- 创建 `gcnet_original_stratified/__init__.py`：导出正式 control 的模型与配置接口。
- 创建 `gcnet_original_stratified/model.py`：无新增参数的 Original `GraphModel` 调用适配器。
- 创建 `gcnet_original_stratified/train_gcnet.py`：仅负责 Original control 的训练损失、stratified 审计、checkpoint 选择和结果写出。
- 创建 `tests/test_original_gcnet_stratified.py`：锁定模型等价、重建目标、预算、hash 与 CLI 合同。
- 修改 `scripts/run_missing_m3_iemocap_current.py`：新增显式 `--model-arm original-gcnet`，默认 Missing-M3 行为不变。
- 修改 `tests/test_missing_m3_iemocap_current_runner.py`：验证 Original 五任务命令和历史默认命令。
- 创建 `experiments/original_gcnet_iemocap6_stratified_20260902/EXPERIMENT.md`：只在正式结果完成后记录协议、审计和三臂对比。

### 任务 1：锁定无参数 Original 模型适配器

**文件：**
- 创建：`tests/test_original_gcnet_stratified.py`
- 创建：`gcnet_original_stratified/__init__.py`
- 创建：`gcnet_original_stratified/model.py`

- [ ] **步骤 1：编写模型等价红灯测试**

在 `tests/test_original_gcnet_stratified.py` 中实例化
`gcnet_modality_jepa.model.GraphModel` 和尚不存在的
`OriginalGCNetControl`。两次构造前重置相同 Torch seed，并验证：

```python
assert control.state_dict().keys() == reference.state_dict().keys()
for key in reference.state_dict():
    torch.testing.assert_close(
        control.state_dict()[key], reference.state_dict()[key], rtol=0, atol=0
    )
assert sum(p.numel() for p in control.parameters()) == sum(
    p.numel() for p in reference.parameters()
)
assert not any(
    token in name
    for name, _ in control.named_parameters()
    for token in ("teacher", "projector", "mmoe", "predictor", "completion")
)
```

在 `eval()` 下对同一完整/缺失输入比较 logits、reconstruction 与
hidden，并对 `logits.square().mean() + recon.square().mean()` 做 backward，
逐项比较 input gradient 和全部 parameter gradient，容差 `1e-6`。

- [ ] **步骤 2：在既有远程测试环境取得正确红灯**

运行：

```bash
scripts/remote_missing_m3.sh sync tests/test_original_gcnet_stratified.py
scripts/remote_missing_m3.sh test tests/test_original_gcnet_stratified.py -q
```

预期：collection error，`gcnet_original_stratified` 尚不存在。不要再探测本地
Python；本仓库已锁定远程测试 Python 为 `/data2/yb/reproduction_envs/s0/bin/python`。

- [ ] **步骤 3：实现最小适配器**

`gcnet_original_stratified/model.py` 的核心为：

```python
class OriginalGCNetControl(GraphModel):
    reconstruction_loss_variant = "corrected-formal-repo"

    def __init__(self, base_model, adim, tdim, vdim, D_e,
                 graph_hidden_size, n_speakers, window_past,
                 window_future, n_classes, dropout=0.5,
                 time_attn=False, no_cuda=False):
        super().__init__(
            base_model, adim, tdim, vdim, D_e, graph_hidden_size,
            n_speakers, window_past, window_future, n_classes,
            dropout, time_attn, no_cuda,
            enable_reconstruction=True,
            graph_branch_mode="both",
            recurrent_padding_mode="legacy",
            postgraph_sequence_mode="independent",
            graph_message_calibration="none",
        )

    def forward(self, inputfeats, availability, qmask, umask,
                seq_lengths, predict_missing=False):
        if predict_missing:
            raise ValueError("Original GCNet has no missing-latent predictor")
        logits, reconstruction, hidden = super().forward(
            inputfeats, qmask, umask, seq_lengths
        )
        return logits, reconstruction, hidden, None
```

`availability` 只用于与共享 evaluator 对齐调用签名，不能参与计算。

- [ ] **步骤 4：同步并验证绿灯**

运行：

```bash
scripts/remote_missing_m3.sh sync gcnet_original_stratified tests/test_original_gcnet_stratified.py
scripts/remote_missing_m3.sh test tests/test_original_gcnet_stratified.py -q
```

预期：模型等价、梯度和 forbidden-parameter 测试全部通过。

- [ ] **步骤 5：提交模型适配器**

仅暂存本任务三个文件，使用 Lore commit，记录适配器必须保持零参数和
state-dict key 等价。

### 任务 2：实现 Original stratified 训练目标与审计

**文件：**
- 修改：`tests/test_original_gcnet_stratified.py`
- 创建：`gcnet_original_stratified/train_gcnet.py`

- [ ] **步骤 1：编写重建目标红灯测试**

从新 trainer 导入 `original_control_loss`。构造两个模态缺失、一个完整和
一个 padding utterance，验证：

```python
total, task, reconstruction = original_control_loss(
    logits=logits,
    reconstruction=[prediction],
    complete_features=complete,
    availability=availability,
    labels=labels,
    umask=umask,
    dataset="IEMOCAPSix",
    dimensions=(2, 3, 1),
)
expected = torch.stack((audio_missing_mse, text_missing_mse)).mean()
torch.testing.assert_close(reconstruction, expected)
torch.testing.assert_close(total, task + expected)
```

再把所有有效 availability 设为 ATV，要求 reconstruction 为可反向传播的
精确零。

- [ ] **步骤 2：编写单 view 训练预算红灯测试**

使用 32 个 fake conversations、8 个 fake schedules 和一个计数模型调用
`train_epoch`。验证：

```python
assert metrics["source_conversation_count"] == 32
assert metrics["masked_view_count"] == 32
assert metrics["model_forward_count"] == 1
assert metrics["optimizer_steps"] == 1
assert metrics["rate_conversation_counts"] == {
    str(rate): 4 for rate in MISSING_RATES
}
assert metrics["jepa_target_count"] == 0
assert metrics["reconstruction_target_count"] == sum(
    metrics["rate_missing_modality_counts"].values()
)
```

同时独立调用 `stratified_rates_for_batch` 重算 expected assignment hash，要求
与 trainer 记录完全相等。

- [ ] **步骤 3：运行新测试确认红灯**

运行：

```bash
scripts/remote_missing_m3.sh sync tests/test_original_gcnet_stratified.py
scripts/remote_missing_m3.sh test tests/test_original_gcnet_stratified.py -q
```

预期：`original_control_loss`、`train_epoch` 或配置接口尚不存在。

- [ ] **步骤 4：实现专用 trainer**

创建冻结 dataclass `OriginalTrainConfig`，只保留 dataset、fold、seed、模型
窗口/hidden/dropout、batch/epoch、optimizer、gradient clip、evaluation
protocol、device 和 `train_rate_mode="stratified"`。拒绝其他 train-rate mode。

复用以下已验证接口，不复制 mask 算法：

```python
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES, STRATIFIED_RATE_ALGORITHM, stratified_rates_for_batch,
)
from gcnet_missing_m3.train_gcnet import (
    _collect_predictions, _metrics, _move_batch, _prepare_stratified_view,
    _prepare_view, _resolve_task_contract, _save_best_checkpoint, _schedules,
    _sha256_tensor, _state_to_cpu, _task_loss, _write_json, evaluate_rate,
    get_loaders, set_random_seed,
)
from gcnet_modality_jepa.loss import MaskedReconLoss
```

每个 batch 必须：确定 rate assignment；构造一个 mixed view；调用一次
`OriginalGCNetControl`；计算 `task + reconstruction`；backward；clip；step。
assignment digest 必须沿用 `b"\0" + assignment_hash` 的现有拼接方式。

`history.json` 每 epoch 记录 task/reconstruction/total loss、raw counts、realized
rate、assignment hash 和训练预算。checkpoint 仍由八率 validation W-F1 均值
选择。最终恢复 `best.pt`，调用共享 `evaluate_rate` 输出八个 test NPZ。

- [ ] **步骤 5：同步并运行训练器单测**

运行：

```bash
scripts/remote_missing_m3.sh sync gcnet_original_stratified tests/test_original_gcnet_stratified.py
scripts/remote_missing_m3.sh test tests/test_original_gcnet_stratified.py -q
```

预期：所有 Original control 单测通过。

- [ ] **步骤 6：提交 trainer**

仅暂存新 trainer 和对应测试，Lore commit 必须写明使用 corrected formal
reconstruction loss，拒绝 literal-upstream flatten bug。

### 任务 3：接入共享 GPU runner 且保持历史默认不变

**文件：**
- 修改：`scripts/run_missing_m3_iemocap_current.py`
- 修改：`tests/test_missing_m3_iemocap_current_runner.py`

- [ ] **步骤 1：编写 runner 红灯测试**

新增：

```python
def test_original_stratified_jobs_use_independent_module(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(2, 3),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
        train_rate_mode="stratified",
        model_arm="original-gcnet",
    )
    assert len(jobs) == 5
    assert all(
        job.command[2:4] == ("-m", "gcnet_original_stratified.train_gcnet")
        for job in jobs
    )
    assert all("--jepa-weight" not in job.command for job in jobs)
```

历史默认测试继续要求 module 为 `gcnet_missing_m3.train_gcnet`、mode 为 `all`
且 JEPA weight 为 0.1。

- [ ] **步骤 2：运行 runner 测试确认红灯**

运行：

```bash
python -m pytest -q tests/test_missing_m3_iemocap_current_runner.py
```

预期：`_build_jobs` 不接受 `model_arm`。

- [ ] **步骤 3：实现显式 arm 选择**

新增 `--model-arm`，choices 为 `missing-m3` 与 `original-gcnet`，默认
`missing-m3`。CLI 的 JEPA weight 默认先保持未解析：Missing-M3 arm 解析为
`0.1`，Original arm 解析为 `0.0`；Original command 只传通用参数并强制
stratified，manifest 记录 arm。显式给 Original arm 传非零 JEPA weight 或
非 stratified mode 时必须拒绝。

- [ ] **步骤 4：运行 runner 与相关回归测试**

运行：

```bash
python -m pytest -q tests/test_missing_m3_iemocap_current_runner.py
scripts/remote_missing_m3.sh sync scripts/run_missing_m3_iemocap_current.py tests/test_missing_m3_iemocap_current_runner.py
scripts/remote_missing_m3.sh test tests/test_missing_m3_iemocap_current_runner.py -q
```

预期：本地无 Torch 的纯 runner 测试和远程同一测试全部通过。

- [ ] **步骤 5：提交 runner**

Lore commit 记录默认 arm 未改变，Original 必须显式选择。

### 任务 4：完整验证与唯一一次远程 smoke

**文件：**
- 验证：`gcnet_original_stratified/`
- 验证：`gcnet_missing_m3/`
- 验证：相关 tests 与 runner

- [ ] **步骤 1：运行相关完整测试套件**

一次同步所有变更后运行：

```bash
scripts/remote_missing_m3.sh sync gcnet_original_stratified gcnet_missing_m3 scripts/run_missing_m3_iemocap_current.py tests/test_original_gcnet_stratified.py tests/test_missing_m3.py tests/test_missing_m3_iemocap_current_runner.py
scripts/remote_missing_m3.sh test tests/test_original_gcnet_stratified.py tests/test_missing_m3.py tests/test_missing_m3_iemocap_current_runner.py -q
```

预期：零失败。不要重复跑相同 smoke 或重新检查 Python 环境。

- [ ] **步骤 2：执行唯一一次 1-epoch GPU2 smoke**

运行到唯一的新目录 `/tmp/gcnet_original_stratified_smoke_20260902_v1`：

```bash
scripts/remote_missing_m3.sh train 2 -m gcnet_original_stratified.train_gcnet \
  --dataset IEMOCAPSix --fold 5 \
  --audio-feature wav2vec-large-c-UTT \
  --text-feature deberta-large-4-UTT --video-feature manet_UTT \
  --output-dir /tmp/gcnet_original_stratified_smoke_20260902_v1 \
  --seed 66 --epochs 1 --batch-size 32 --train-rate-mode stratified \
  --hidden 200 --windowp 2 --windowf 2 --lr 0.001 --l2 0.00001 \
  --dropout 0.5 --gradient-clip-norm 1.0 --evaluation-protocol official
```

核验 120 source/120 view、4 forward/4 update、每 rate 15 conversations、有限
task/reconstruction/total loss、八率 validation/test 和 8 NPZ。

- [ ] **步骤 3：匹配 candidate hash**

对 smoke epoch 0，使用同 seed/fold/batch order 重算 assignment hash，并与
With-JEPA seed 66 epoch 1 的 hash 比较。测试 schedule 的八个 mask hash 也必须
逐项相同；任一不同时停止正式队列并修复，不得解释为随机波动。

- [ ] **步骤 4：独立代码与协议复核**

调度一个只读 reviewer 检查参数纯度、loss、预算和 provenance。发现严重或
中等问题时先修复并只重跑受影响测试；PASS 后才能开始正式实验。

### 任务 5：运行五种子正式 Original control 并形成三臂结论

**文件：**
- 创建：`experiments/original_gcnet_iemocap6_stratified_20260902/EXPERIMENT.md`
- 创建：`experiments/original_gcnet_iemocap6_stratified_20260902/results/`

- [ ] **步骤 1：启动五任务队列**

在 healthy GPU 2/3 上运行，每卡最多三个进程，禁止 GPU4：

```bash
python -u scripts/run_missing_m3_iemocap_current.py \
  --repo-root /data2/yb/paper/GCNet_TPAMI_single_view_dev \
  --output-root /data2/yb/remote_experiments/original_gcnet_iemocap6_stratified_20260902/formal \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python \
  --gpus 2 3 --max-concurrent-per-gpu 3 \
  --datasets IEMOCAPSix --model-arm original-gcnet \
  --train-rate-mode stratified
```

- [ ] **步骤 2：审计 5/5 jobs**

核验 500 epochs、40 NPZ、无失败/NaN/类别坍塌；每个 epoch 为 120 source、
120 views、4 forwards/updates、每 rate 15 conversations；逐 seed 对齐 With-JEPA
和 gradient-off 的 100 个 assignment hashes、八个 test mask hashes、labels 和
availability。

- [ ] **步骤 3：计算三臂结果**

输出每 seed×rate W-F1、五种子均值/标准差、八率 mean、高率 0.4--0.7 mean、
paired deltas、positive seeds、95% t interval。分别回答：

```text
With-JEPA - JEPA-gradient-off = JEPA gradient contribution
With-JEPA - matched Original = full method comparison
JEPA-gradient-off - matched Original = architecture-bundle comparison
```

只有证据支持时才写稳定或显著；否则保留方向性结论。

- [ ] **步骤 4：复制轻量结果并提交**

同步 config/history/metrics/status/NPZ/runner manifest/log，不同步 `best.pt`。
运行 JSON、NPZ 指标复算、SHA、`git diff --check`，并确认用户的
`experiments/missing_m3_mosei_current_20260831/` 未暂存、未修改。

- [ ] **步骤 5：推送 GitHub**

使用 Lore commit，推送 `github/feature/missing-m3-sdr-backbone`，确认远端 SHA
等于本地 HEAD 后再报告最终结论。
