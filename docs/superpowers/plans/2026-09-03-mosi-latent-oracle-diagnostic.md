# MOSI Missing-Latent Oracle 诊断实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用已有 Slot Missing-M3 checkpoints，在 CMU-MOSI validation 的 8 个 missing rates 上完成 graph-only、predicted、real-teacher、shuffled-teacher 四路冻结推理诊断。

**架构：** 新增独立诊断库和 CLI，不修改模型与训练器。CLI 从历史 checkpoint 重建相同 loader 和 mask，将有效 utterance 按 MOSI 指标顺序展平，再复用原 fusion 与 classifier 计算四路结果。

**技术栈：** Python 3.8、PyTorch 1.8、NumPy、scikit-learn、pytest、现有 GCNet/Missing-M3 模块。

---

## 文件结构

- 创建 `gcnet_missing_m3/oracle_diagnostic.py`：纯张量诊断操作、确定性 target-wise shuffle、effective rank 与模型状态哈希。
- 创建 `scripts/run_mosi_latent_oracle.py`：checkpoint/loader 构建、validation 收集、四路评估和结果写入。
- 创建 `tests/test_mosi_latent_oracle.py`：锁定 shuffle、flatten、fusion、状态不变和指标重算合同。
- 创建 `experiments/missing_m3_mosi_latent_oracle_20260903/EXPERIMENT.md`：协议、命令、结果与结论。
- 生成 `experiments/missing_m3_mosi_latent_oracle_20260903/results/`：JSON、NPZ 和汇总。

### 任务 1：纯诊断操作

**文件：**
- 创建：`tests/test_mosi_latent_oracle.py`
- 创建：`gcnet_missing_m3/oracle_diagnostic.py`

- [x] **步骤 1：编写确定性 shuffle 与 flatten 的失败测试**

```python
def row_multiset(value):
    return sorted(tuple(row.tolist()) for row in value)


def test_shuffle_is_target_specific_and_deterministic():
    shuffled_a = shuffle_missing_targets(teacher, target_mask, seed=17)
    shuffled_b = shuffle_missing_targets(teacher, target_mask, seed=17)
    assert torch.equal(shuffled_a, shuffled_b)
    for target in range(3):
        selected = target_mask[:, target]
        assert row_multiset(shuffled_a[selected, target]) == row_multiset(
            teacher[selected, target]
        )
    assert torch.equal(shuffled_a[~target_mask], teacher[~target_mask])


def test_flatten_valid_uses_batch_major_metric_order():
    assert torch.equal(
        flatten_valid(values, umask),
        values.transpose(0, 1)[umask.bool()],
    )
```

- [x] **步骤 2：运行测试，确认因接口不存在而失败**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  tests/test_mosi_latent_oracle.py -q
```

预期：collection error，缺少 `gcnet_missing_m3.oracle_diagnostic`。

- [x] **步骤 3：实现最少纯函数**

```python
def flatten_valid(value, umask):
    return value.transpose(0, 1)[umask.bool()]


def stable_seed(*parts):
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def shuffle_missing_targets(teacher, target_mask, seed):
    output = teacher.clone()
    for target in range(teacher.shape[1]):
        selected = torch.nonzero(target_mask[:, target], as_tuple=False).flatten()
        if selected.numel() > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_seed(seed, target))
            order = torch.randperm(selected.numel(), generator=generator)
            shift = 1 + stable_seed(seed, target, "shift") % (selected.numel() - 1)
            sources = order.roll(int(shift))
            output[selected[order], target] = teacher[selected[sources], target]
    return output
```

同一文件实现 `effective_rank(value)`、`tensor_sha256(value)`、`state_dict_sha256(model)`、named-buffer snapshot/restore 和 conversation-cluster paired bootstrap；哈希必须包含 key、dtype、shape 与连续 tensor bytes。shuffle 测试必须断言有效池的 permutation fixed-point count 为 0。

- [x] **步骤 4：运行 focused tests，确认通过**

预期：所有任务 1 测试通过，且没有修改任何核心模型文件。

- [x] **步骤 5：提交纯诊断操作**

提交信息遵循 Lore Commit Protocol，并记录 focused test 结果。

### 任务 2：四路冻结推理

**文件：**
- 修改：`tests/test_mosi_latent_oracle.py`
- 修改：`gcnet_missing_m3/oracle_diagnostic.py`
- 创建：`scripts/run_mosi_latent_oracle.py`

- [x] **步骤 1：编写四路输出的失败测试**

```python
def test_predicted_path_matches_native_completion_forward():
    native_logits, native_hidden, _, _ = model(
        [incomplete], availability, qmask, umask, lengths,
        predict_missing=False,
    )
    paths = compute_oracle_paths(
        model, incomplete, complete, availability, qmask, umask, lengths,
    )
    assert torch.allclose(paths["predicted"], native_logits, atol=1e-6)
    assert torch.allclose(paths["predicted_hidden"], native_hidden, atol=1e-6)


def test_graph_only_bypasses_fusion_bias():
    paths = compute_oracle_paths(
        model,
        incomplete,
        complete,
        availability,
        qmask,
        umask,
        lengths,
    )
    assert torch.equal(paths["graph_only"], model.smax_fc(paths["graph_hidden"]))
```

另加测试：teacher 只填 `target_mask`、所有输出 finite、运行前后 state hash 相同、非持久 routing buffers 被恢复、rate 0 四路等价、conversation bootstrap 不拆分同一对话。

- [x] **步骤 2：运行新测试，确认因 `compute_oracle_paths` 缺失而失败**

- [x] **步骤 3：实现一次 batch 收集与全 split 四路计算**

核心数据流固定为：

```python
encoded, latents = model.observed_set(incomplete, availability, umask)
graph = model.encode_hidden([encoded], qmask, umask, lengths)
pred = model.missing_predictor(latents, graph, availability, umask)
teacher_dict = model.encode_teacher_targets([complete])
teacher = torch.stack([teacher_dict[name] for name in MODALITIES], dim=2)
```

先将 graph、prediction、teacher、target mask、label 和 availability 全部按 `flatten_valid` 收集，并由 conversation ID 与 utterance index 生成 sample key。随后统一构造成 `[N,1,3,D]` 调用原 `missing_latent_fusion`，避免 batch 内 shuffle 和样本顺序错位。测试还必须覆盖全 split shuffle 对 batch 分割方式不敏感，以及修改 complete target 不会改变 predicted 路径。

- [x] **步骤 4：实现 CLI 与输出 schema**

```bash
python scripts/run_mosi_latent_oracle.py \
  --feature-root /path/to/CMUMOSI/features \
  --checkpoint-root /path/to/formal \
  --history-root experiments/missing_m3_mosi_classification_completion_20260829/results/formal \
  --output-dir experiments/missing_m3_mosi_latent_oracle_20260903/results \
  --seeds 66 67 68 69 70 \
  --rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --shuffle-count 8 \
  --split validation \
  --device cuda
```

CLI 必须拒绝非 `CMUMOSI`、非 validation split、缺少 completion keys、配置与 checkpoint 不一致，以及覆盖非空结果目录。结果还需记录 raw latent、target-specific LayerNorm 后 latent、Linear 投影前后 RMS、tanh 饱和比例、fusion residual 和相对 graph logit delta 的统计，区分表示 OOD 与分类效果。

- [x] **步骤 5：运行 focused 与历史回归测试**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  tests/test_mosi_latent_oracle.py tests/test_missing_m3.py -q
```

预期：全部通过；原 `tests/test_missing_m3.py` 保持 60/60 通过。

- [x] **步骤 6：提交 runner**

提交信息记录 predicted/native 等价阈值、状态哈希与未运行的正式 validation。

CLI 中所有 NPZ 行都使用 conversation-major 顺序；另用历史 time-major 顺序计算 `legacy_mask_sha256`，使其可与原 `metrics.json` 核对。

### 任务 3：远程 validation 诊断与证据汇总

**文件：**
- 创建：`experiments/missing_m3_mosi_latent_oracle_20260903/EXPERIMENT.md`
- 生成：`experiments/missing_m3_mosi_latent_oracle_20260903/results/summary.json`
- 生成：`experiments/missing_m3_mosi_latent_oracle_20260903/results/seed_*/rate_*.json`
- 生成：`experiments/missing_m3_mosi_latent_oracle_20260903/results/seed_*/rate_*.npz`

- [x] **步骤 1：同步隔离分支到 biggpu 并运行完整测试**

使用远端既有 `s0` pytest 环境运行单元测试；正式推理使用 `gcnet-official`。不得安装依赖，不得使用 GPU 4。

- [x] **步骤 2：运行 5 seeds × 8 rates × 4 paths 的冻结 validation 推理**

使用 `/data2/yb/reproduction_envs/gcnet-official/bin/python3.8`。该步骤不创建 optimizer、不调用 backward、不保存新 checkpoint。

- [x] **步骤 3：验证 provenance 与结果完整性**

检查：

```text
40/40 seed-rate JSON
40/40 seed-rate NPZ
predicted/native max_abs_error < 1e-6
40/40 model state SHA before == after
所有 prediction finite
所有 checkpoint SHA 已记录
所有 legacy mask SHA 与 aligned availability SHA 已记录
所有 sample-order SHA 与 target-wise permutation SHA 已记录
rate 0.0 四路 max_abs_error < 1e-6
所有 named buffers 在每个 rate 后已恢复
```

- [x] **步骤 4：写实验结论**

以 `real_teacher - shuffled_teacher` 作为样本级信息的主对照，并报告 7 个非零 rate 的均值与 0.5--0.7 高缺失均值。`real_teacher - predicted` 只作辅助量，因为它受 teacher-to-fusion 分布外输入混杂。使用 conversation-cluster bootstrap 报告 paired W-F1 delta 的 95% 区间。只有五种子配对效应为正且区间排除 0 时称为强证据；方向一致但区间包含 0 时称为弱证据。每个 seed/rate 先运行 8 个 shuffle；若 shuffle W-F1 的 Monte Carlo 标准误差超过 0.1 个百分点，则自动扩展到 32 个。若 real-teacher 相对 shuffle 没有稳定增量，则记录「直接注入不能区分 teacher 信息与 fusion OOD」，并把冻结 supervised probe 作为独立后续诊断，不修改本轮结论。即便 real-teacher 显著更好，结论也只能是「privileged missing-modal information 有用但当前 predicted residual 未提供」，不能声称现有 observed sources 必然可恢复该信息。

- [x] **步骤 5：提交结果并推送实验分支**

Lore commit 必须记录实际命令、通过数、checkpoint 数、结果文件数及未测试的 test split。
