# Equal-Budget Stratified Missing-Rate 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Missing-M3 增加 conversation-level batch-balanced random `stratified` 训练模式，使每段对话每个 epoch 只产生一个 masked view，同时在 batch 内均衡覆盖请求缺失率 0.0--0.7。

**架构：** 在 `mixed_rate.py` 中实现无全局 RNG 的确定性分层分配器；在 Missing-M3 trainer 内按 conversation 选择已有 `ConversationMaskSchedule` 并组装一个混合 rate batch；训练生命周期仍然只有一次 batched forward/backward/optimizer step。历史 `fixed`、`cyclic`、`all` 路径保持原语义，正式 runner 显式选择新模式。

**技术栈：** Python 3、PyTorch、NumPy、pytest、现有 GCNet/Missing-M3 trainer、SHA-256 provenance。

---

## 文件职责

- 修改 `gcnet_missing_m3/mixed_rate.py`：纯 Python rate 分层分配、确定性 shuffle 和 assignment hash。
- 修改 `gcnet_missing_m3/train_gcnet.py`：按 conversation 构造 mixed-rate mask、接入单 view 训练生命周期、记录预算与 realized-rate 审计。
- 修改 `scripts/run_missing_m3_iemocap_current.py`：显式转发 `--train-rate-mode`，默认仍为 `all`，避免历史命令漂移。
- 修改 `tests/test_missing_m3.py`：分配器、mask 对齐、训练生命周期、CLI 和旧模式回归测试。
- 修改 `tests/test_missing_m3_iemocap_current_runner.py`：runner 的新模式转发测试。
- 创建 `experiments/missing_m3_iemocap6_stratified_jepa_ablation_20260902/EXPERIMENT.md`：正式实验协议与最终结论，仅在结果完成后写入结果。

### 任务 1：锁定分层 rate 分配器

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/mixed_rate.py`

- [ ] **步骤 1：编写失败的纯函数测试**

在 `tests/test_missing_m3.py` 增加以下测试，导入尚不存在的
`StratifiedRateAssignment` 与 `stratified_rates_for_batch`：

```python
def test_stratified_rates_balance_full_and_partial_batches():
    full = stratified_rates_for_batch(
        rates=MISSING_RATES,
        master_seed=66,
        dataset="IEMOCAPSix",
        fold=5,
        epoch=0,
        batch_index=0,
        epoch_size=52,
        conversations_seen=0,
        conversation_ids=tuple(f"c{i}" for i in range(32)),
    )
    assert len(full.rates) == 32
    assert {rate: full.rates.count(rate) for rate in MISSING_RATES} == {
        rate: 4 for rate in MISSING_RATES
    }

    tail = stratified_rates_for_batch(
        rates=MISSING_RATES,
        master_seed=66,
        dataset="IEMOCAPSix",
        fold=5,
        epoch=0,
        batch_index=1,
        epoch_size=52,
        conversations_seen=32,
        conversation_ids=tuple(f"tail{i}" for i in range(20)),
    )
    counts = [tail.rates.count(rate) for rate in MISSING_RATES]
    assert max(counts) - min(counts) == 1
    assert set(tail.rates) == set(MISSING_RATES)
```

再增加三个不变量测试：

```python
def test_stratified_rates_are_deterministic_seeded_and_rng_isolated():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    kwargs = dict(
        rates=MISSING_RATES,
        master_seed=66,
        dataset="CMUMOSI",
        fold=1,
        epoch=3,
        batch_index=0,
        epoch_size=52,
        conversations_seen=0,
        conversation_ids=tuple(f"v{i}" for i in range(32)),
    )
    first = stratified_rates_for_batch(**kwargs)
    second = stratified_rates_for_batch(**kwargs)
    changed = stratified_rates_for_batch(**{**kwargs, "master_seed": 67})
    assert first == second
    assert first.rates != changed.rates
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    assert np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_stratified_tail_surplus_continues_across_epochs():
    # N=20 gives counts 2/3 per epoch. Across two epochs every rate must receive
    # exactly five assignments because the second epoch begins at offset 4.
    assignments = [
        stratified_rates_for_batch(
            rates=MISSING_RATES,
            master_seed=66,
            dataset="CMUMOSI",
            fold=1,
            epoch=epoch,
            batch_index=0,
            epoch_size=20,
            conversations_seen=0,
            conversation_ids=tuple(f"v{i}" for i in range(20)),
        )
        for epoch in (0, 1)
    ]
    combined = assignments[0].rates + assignments[1].rates
    assert [combined.count(rate) for rate in MISSING_RATES] == [5] * 8


def test_stratified_small_tail_never_duplicates_conversations():
    assignment = stratified_rates_for_batch(
        rates=MISSING_RATES,
        master_seed=66,
        dataset="CMUMOSI",
        fold=1,
        epoch=0,
        batch_index=0,
        epoch_size=5,
        conversations_seen=0,
        conversation_ids=("a", "b", "c", "d", "e"),
    )
    assert len(assignment.rates) == 5
    assert len(set(assignment.rates)) == 5
```

- [ ] **步骤 2：运行测试确认红灯**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified_rates'
```

预期：collection error 或 import error，指出新接口尚不存在。

- [ ] **步骤 3：实现最小纯函数**

在 `gcnet_missing_m3/mixed_rate.py` 增加：

```python
STRATIFIED_RATE_ALGORITHM = "conversation-rate-stratified-v1"


@dataclass(frozen=True)
class StratifiedRateAssignment:
    rates: tuple[float, ...]
    assignment_hash: str
    algorithm: str = STRATIFIED_RATE_ALGORITHM


def stratified_rates_for_batch(
    rates: Sequence[float],
    *,
    master_seed: int,
    dataset: str,
    fold: int | str,
    epoch: int,
    batch_index: int,
    epoch_size: int,
    conversations_seen: int,
    conversation_ids: Sequence[str],
) -> StratifiedRateAssignment:
    normalized = tuple(float(rate) for rate in rates)
    identifiers = tuple(str(value) for value in conversation_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("rates must be nonempty and unique")
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("conversation_ids must be nonempty strings")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if epoch < 0 or batch_index < 0 or conversations_seen < 0:
        raise ValueError("epoch and stream positions must be nonnegative")
    batch_size = len(identifiers)
    if epoch_size < batch_size or conversations_seen + batch_size > epoch_size:
        raise ValueError("batch positions exceed epoch_size")
    quotient, remainder = divmod(batch_size, len(normalized))
    stream_offset = epoch * epoch_size + conversations_seen
    balanced = list(normalized) * quotient
    balanced.extend(
        normalized[(stream_offset + index) % len(normalized)]
        for index in range(remainder)
    )
    payload = {
        "algorithm": STRATIFIED_RATE_ALGORITHM,
        "master_seed": master_seed,
        "dataset": dataset,
        "fold": fold,
        "epoch": epoch,
        "batch_index": batch_index,
        "epoch_size": epoch_size,
        "conversations_seen": conversations_seen,
        "conversation_ids": list(identifiers),
        "rates": [format(rate, ".17g") for rate in normalized],
    }
    payload_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload_bytes).digest()[:8], "big")
    local_rng = random.Random(local_seed)
    local_rng.shuffle(balanced)
    assigned = tuple(balanced)
    assigned_bytes = json.dumps(
        [format(rate, ".17g") for rate in assigned],
        separators=(",", ":"),
    ).encode("ascii")
    assignment_hash = hashlib.sha256(
        payload_bytes + b"\0" + assigned_bytes
    ).hexdigest()
    return StratifiedRateAssignment(assigned, assignment_hash)
```

Use a local `random.Random`; never call global `random.shuffle`, `np.random`,
`torch.rand`, or Python `hash()`.

- [ ] **步骤 4：运行分配器测试确认绿灯**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified_rates or balanced_rate_schedule'
```

预期：新增测试与原 cyclic schedule 测试全部通过。

- [ ] **步骤 5：提交分配器**

仅暂存 `gcnet_missing_m3/mixed_rate.py` 与 `tests/test_missing_m3.py`，使用
Lore commit，记录尾 batch 不复制和无全局 RNG 的约束。

### 任务 2：按 conversation 构造一个 mixed-rate view

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写 mask 对齐红灯测试**

构造两个 conversation、不同长度和 speaker `qmask`，为第一个分配
`eta=0.0`、第二个分配 `eta=0.7`。用 fake schedule 的 `generate()` 返回
可辨识的 host/guest mask，然后验证：

```python
host, guest = train_gcnet._build_stratified_mask_tensors(
    schedules={0.0: complete_schedule, 0.7: sparse_schedule},
    conversation_rates=(0.0, 0.7),
    conversation_ids=("c0", "c1"),
    umask=umask,
    epoch=4,
)
assert host.shape == guest.shape == (sequence_length, 2, 3)
assert torch.equal(host[:, 0], complete_expected)
assert torch.equal(host[:, 1], sparse_expected)
assert torch.equal(guest[:, 0], complete_guest_expected)
assert not host[padding_positions].any()
assert not guest[padding_positions].any()
```

再测试 rate 数量、ID 数量、schedule key 不匹配均抛出 `ValueError`。

- [ ] **步骤 2：运行测试确认红灯**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified_mask'
```

预期：FAIL，`_build_stratified_mask_tensors` 不存在。

- [ ] **步骤 3：实现本地 mask helper 和 view helper**

在 `gcnet_missing_m3/train_gcnet.py` 增加：

```python
def _build_stratified_mask_tensors(
    schedules: Mapping[float, ConversationMaskSchedule],
    conversation_rates: Sequence[float],
    conversation_ids: Sequence[str],
    umask: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if umask.ndim != 2:
        raise ValueError("umask must have shape [batch, sequence]")
    identifiers = tuple(str(value) for value in conversation_ids)
    rates = tuple(float(value) for value in conversation_rates)
    batch_size, sequence_length = umask.shape
    if len(identifiers) != batch_size or len(rates) != batch_size:
        raise ValueError("rates, IDs, and umask batch size must match")
    if any(rate not in schedules for rate in rates):
        raise ValueError("conversation rate has no mask schedule")
    side_tensors = []
    for side in ("host", "guest"):
        conversations = []
        for index, (conversation_id, rate) in enumerate(zip(identifiers, rates)):
            valid_length = int(umask[index].sum().item())
            if valid_length < 1:
                raise ValueError("every conversation needs a valid utterance")
            generated = schedules[rate].generate(
                conversation_id,
                length=sequence_length,
                valid_length=valid_length,
                side=side,
                epoch=epoch,
            )
            conversations.append(
                torch.as_tensor(generated.availability, device=umask.device)
            )
        side_tensors.append(torch.stack(conversations, dim=1))
    return side_tensors[0], side_tensors[1]
```

提取仅负责复用现有 feature/mask 拼装的 helper：

```python
def _prepare_view_from_primary_masks(
    data, host_availability, guest_availability, dimensions
) -> dict[str, object]:
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    qmask, umask, labels = data[6], data[7], data[8]
    full = generate_inputs(
        audio_host, text_host, visual_host,
        audio_guest, text_guest, visual_guest, qmask,
    )[0]
    availability = generate_inputs(
        host_availability[..., 0:1],
        host_availability[..., 1:2],
        host_availability[..., 2:3],
        guest_availability[..., 0:1],
        guest_availability[..., 1:2],
        guest_availability[..., 2:3],
        qmask,
    )[0].to(dtype=full.dtype)
    expanded = torch.repeat_interleave(
        availability,
        torch.tensor(dimensions, device=availability.device),
        dim=-1,
    )
    return {
        "complete": full,
        "incomplete": full * expanded,
        "availability": availability,
        "qmask": qmask,
        "umask": umask,
        "labels": labels,
        "lengths": _lengths(umask),
        "conversation_ids": list(data[-1]),
    }


def _prepare_stratified_view(
    data, schedules, conversation_rates, epoch, dimensions
) -> dict[str, object]:
    host, guest = _build_stratified_mask_tensors(
        schedules=schedules,
        conversation_rates=conversation_rates,
        conversation_ids=data[-1],
        umask=data[7],
        epoch=epoch,
    )
    return _prepare_view_from_primary_masks(data, host, guest, dimensions)
```

原 `_prepare_view(data, schedule, epoch, dimensions)` 继续通过原
`build_primary_mask_tensors()` 调用同一拼装 helper，签名不变。

- [ ] **步骤 4：增加真实 schedule 不变量测试并跑绿**

使用真实 `ConversationMaskSchedule` 验证：

- `eta=0` conversation 的有效行 bit-exact 等于 `[1,1,1]`；
- 所有其他有效行 `sum(-1)>=1`；
- padding 全零；
- realized missing fraction 由最终 availability 重算。

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified_mask or prepare_view'
```

预期：全部通过。

- [ ] **步骤 5：提交 mixed-rate mask 路径**

仅暂存 trainer 与对应测试，Lore commit 明确共享 mask helper 未修改。

### 任务 3：接入单 view 训练生命周期和预算审计

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写生命周期红灯测试**

为 fake loader 提供 `.dataset` 长度 32 和一个含 32 个 conversation IDs 的
batch；monkeypatch `_prepare_stratified_view` 返回可识别 mixed availability。
验证：

```python
assert model.forward_calls == 1
assert optimizer.zero_grad_calls == 1
assert len(optimizer.step_gradients) == 1
assert model.teacher_calls == 1
assert model.ema_calls == 1
assert metrics["source_conversation_count"] == 32
assert metrics["masked_view_count"] == 32
assert metrics["model_forward_count"] == 1
assert metrics["rate_conversation_counts"] == {
    str(rate): 4 for rate in MISSING_RATES
}
assert metrics["stratified_assignment_hash"]
```

另加测试以 `TrainConfig(train_rate_mode="stratified",
jepa_rate_weighting="sparsity-budget")` 调用与上面相同的完整 fake
`train_epoch` 参数，并断言抛出匹配 `stratified.*uniform` 的 `ValueError`。

- [ ] **步骤 2：运行生命周期测试确认红灯**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified_train or stratified_rejects'
```

预期：FAIL，因为 trainer 尚不认识 `stratified`。

- [ ] **步骤 3：实现 train_epoch 分支**

在 epoch 开始时初始化：

```python
epoch_size = len(loader.dataset)
conversations_seen = 0
source_conversation_count = 0
masked_view_count = 0
model_forward_count = 0
rate_conversation_counts = {rate: 0 for rate in MISSING_RATES}
realized_missing = {rate: [0, 0] for rate in MISSING_RATES}
assignment_digest = hashlib.sha256()
```

新分支只创建一个 view：

```python
elif config.train_rate_mode == "stratified":
    ids = tuple(str(value) for value in data[-1])
    assignment = stratified_rates_for_batch(
        rates=MISSING_RATES,
        master_seed=config.seed,
        dataset=config.dataset,
        fold=config.fold,
        epoch=epoch,
        batch_index=batch_index,
        epoch_size=epoch_size,
        conversations_seen=conversations_seen,
        conversation_ids=ids,
    )
    view = _prepare_stratified_view(
        data, schedules, assignment.rates, epoch, dimensions
    )
    optimizer.zero_grad(set_to_none=True)
    rate_views = ((None, view),)
```

对 `rate is None` 使用 `jepa_rate_weight=1.0`。按 conversation rate 和
`umask` 累计 requested counts、最终 availability 的 missing/total 元素，
并将 assignment hash 串接进 epoch digest。其余 forward、teacher、loss、
backward、clip、step、EMA 顺序保持现有单 view 路径。

返回指标增加：

```python
{
    "source_conversation_count": source_conversation_count,
    "masked_view_count": masked_view_count,
    "model_forward_count": model_forward_count,
    "rate_conversation_counts": {
        str(rate): rate_conversation_counts[rate] for rate in MISSING_RATES
    },
    "rate_realized_missing_fraction": {
        str(rate): (
            realized_missing[rate][0] / realized_missing[rate][1]
            if realized_missing[rate][1] else None
        )
        for rate in MISSING_RATES
    },
    "stratified_assignment_hash": assignment_digest.hexdigest(),
    "stratified_rate_algorithm": STRATIFIED_RATE_ALGORITHM,
}
```

非 stratified 模式保留现有 `rate_batch_counts` 数值；新增通用预算字段可以
记录其真实预算，但不能更改旧字段语义。

- [ ] **步骤 4：扩展协议和 CLI 验证**

使 `_protocol_rates()` 对 `stratified` 返回全部八 rates；CLI choices 接受
`stratified`，默认仍为 `cyclic`。`_fixed_missing_rate()` 继续拒绝在非 fixed
模式传 `fixed_missing_rate`。

- [ ] **步骤 5：运行目标与完整 trainer 测试**

运行：

```bash
pytest -q tests/test_missing_m3.py -k 'stratified or train_rate_mode or protocol_rates'
pytest -q tests/test_missing_m3.py
```

预期：目标测试和 `tests/test_missing_m3.py` 全部通过。

- [ ] **步骤 6：提交训练生命周期**

Lore commit 记录 `stratified` 只支持 uniform JEPA weighting 的范围。

### 任务 4：让现有正式 runner 显式选择协议

**文件：**
- 修改：`tests/test_missing_m3_iemocap_current_runner.py`
- 修改：`scripts/run_missing_m3_iemocap_current.py`

- [ ] **步骤 1：编写 runner 红灯测试**

增加：

```python
def test_stratified_job_matrix_forwards_equal_budget_mode(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(2, 3, 6, 7),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.1,
        train_rate_mode="stratified",
    )
    assert len(jobs) == 5
    assert {
        _argument(job.command, "--train-rate-mode") for job in jobs
    } == {"stratified"}
```

保留默认 runner 仍生成 `all` 的回归断言。

- [ ] **步骤 2：运行测试确认红灯**

运行：

```bash
pytest -q tests/test_missing_m3_iemocap_current_runner.py
```

预期：FAIL，`_build_jobs()` 不接受 `train_rate_mode`。

- [ ] **步骤 3：实现最小 runner 参数**

为 `_build_jobs()` 和 CLI 增加 `train_rate_mode`，choices 为四种现有/新模式，
默认值 `all` 以保持历史 runner 行为。manifest 顶层写入该字段，命令使用：

```python
"--train-rate-mode", train_rate_mode,
```

- [ ] **步骤 4：运行 runner 与回归测试**

运行：

```bash
pytest -q tests/test_missing_m3_iemocap_current_runner.py
pytest -q tests/test_missing_m3.py -k 'train_rate_mode_cli'
```

预期：全部通过。

- [ ] **步骤 5：提交 runner**

Lore commit 记录默认 `all` 仅为历史兼容，正式公平实验必须显式传
`--train-rate-mode stratified`。

### 任务 5：本地与远程一次性验证

**文件：**
- 不新增源文件。

- [ ] **步骤 1：运行相关完整测试集合**

```bash
pytest -q \
  tests/test_missing_m3.py \
  tests/test_missing_m3_iemocap_current_runner.py \
  tests/test_mask_schedule.py \
  tests/test_trainer_protocol_integration.py
git diff --check
```

预期：零失败、零 whitespace error。

- [ ] **步骤 2：代码审查**

检查：旧模式 diff 路径、global RNG、conversation/utterance 计数、tail batch、
padding、speaker host/guest 对齐、assignment hash、requested/realized rate 和
checkpoint/result config。

- [ ] **步骤 3：同步远程并做单个 1-epoch GPU smoke**

只同步本次修改文件到 `/data2/yb/paper/GCNet_TPAMI_single_view_dev`，核对
SHA256。使用官方环境和健康 GPU 运行一个 IEMOCAP-6 seed 66、1 epoch、
`stratified` 任务；验证 history 中：

- 4 个 optimizer steps；
- 4 个 model forwards，而不是 32 个；
- source conversation count 等于 masked-view count；
- 每个完整 batch 每 rate 恰好 4 个 conversation；
- 结果、loss 和梯度有限。

smoke 只验证集成，不用于汇报模型分数，且只运行一次。

- [ ] **步骤 4：提交并推送实现**

使用完成前验证证据写 Lore commit，推送
`feature/missing-m3-sdr-backbone`；不提交 checkpoint。

### 任务 6：公平协议的最小判别实验

**文件：**
- 创建：`experiments/missing_m3_iemocap6_stratified_jepa_ablation_20260902/EXPERIMENT.md`
- 创建：`experiments/missing_m3_iemocap6_stratified_jepa_ablation_20260902/results/SUMMARY.json`

- [ ] **步骤 1：锁定 10 个任务**

IEMOCAP-6 fold 5，seeds 66--70，两个 arms：

```text
With-JEPA:          train_rate_mode=stratified, jepa_weight=0.1
JEPA-gradient-off:  train_rate_mode=stratified, jepa_weight=0.0
```

其他参数完全继承当前 Slot Missing-M3。两臂共享 formal seed、rate assignment、
训练 mask schedule、验证/测试 mask 和选模规则。Original 暂不运行。

- [ ] **步骤 2：远程启动且不使用 GPU4**

先 dry-run 核对恰好 10 个命令，再在健康 GPU 上启动。每张卡的并发数由
实时显存决定，但不得改变任务参数。runner 每个任务只允许一个进程。

- [ ] **步骤 3：完成后独立审计**

要求：10/10 jobs、100 epochs、80 prediction NPZ、两臂 40/40 assignment hash、
mask hash、labels 和 availability 配对；独立重算 W-F1/Macro-F1/Accuracy；
确认每 epoch source conversation count 等于 masked-view count且不含八倍扩张。

- [ ] **步骤 4：执行预注册门槛**

计算 With-JEPA minus gradient-off 的八率和 0.4--0.7 paired seed aggregates。
门槛保持：两类 mean delta 均为正且各至少 3/5 seeds 为正。失败则停止扩展；
通过才进入 stratified Original 控制和其他数据集。

- [ ] **步骤 5：保存轻量结果并推送**

保存 config/history/metrics/log/NPZ/manifest/SUMMARY/EXPERIMENT，排除
`best.pt`。文档明确 `all` 结果仅为额外 view-budget 消融。
