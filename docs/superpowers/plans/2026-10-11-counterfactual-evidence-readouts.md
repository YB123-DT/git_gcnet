# Counterfactual Evidence Readouts 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 cfg84 OSRAM、原 Flat anchor 或任务损失的前提下，实现 CEMR 与 CIR 两个证据干预读出，并完成 MOSI seed66、150 epoch 的单卡双实验启动。

**架构：** 新模块复用原 `emotion_adapter` 的参数，以共享 Dropout mask 计算同一 utterance 的多个 evidence-subset Flat view。CEMR 编码四个 leave-one-out 边际 token；CIR 编码四个 singleton 与六个 pair Möbius token；两者均通过零初始化 residual projector 接回原 Flat pre-norm anchor。

**技术栈：** PyTorch、pytest、现有 `TrainConfig`/OSRAM runner、biggpu V100 32GB、完整训练状态恢复。

---

## 文件结构

- 创建：`gcnet_missing_m3/counterfactual_readout.py` —— 共享 Flat-view 执行器、CEMR、CIR。
- 修改：`gcnet_missing_m3/osram.py` —— 构建模块并在 Flat readout 后调用。
- 修改：`gcnet_missing_m3/model.py` —— 透传新配置。
- 修改：`gcnet_missing_m3/train_gcnet.py` —— 配置验证、CLI、模型构建和运行配置记录。
- 创建：`tests/test_counterfactual_readout.py` —— 数学、mask、Dropout、梯度和兼容性测试。
- 创建：`tests/test_counterfactual_readout_runner.py` —— 训练协议与 runner 合同测试。
- 创建：`experiments/osram_counterfactual_readouts_20261011/run.py` —— A/B 两任务的 sealed runner、smoke、恢复和汇总。
- 创建：`experiments/osram_counterfactual_readouts_20261011/RESULT.md` —— 协议、状态和最终结果。
- 创建：`experiments/osram_counterfactual_readouts_20261011/LAUNCH.md` —— 服务器、GPU、PID、源码和输出记录。

### 任务 1：锁定 Flat-view 数学与随机性

**文件：**
- 创建：`tests/test_counterfactual_readout.py`
- 创建：`gcnet_missing_m3/counterfactual_readout.py`

- [ ] **步骤 1：编写失败的共享视图测试**

```python
def test_equal_views_share_training_dropout_and_rng_advance():
    adapter = flat_adapter(12, 10, dropout=.5).train()
    views = torch.randn(3, 2, 1, 12).expand(-1, -1, 5, -1).clone()
    state = torch.get_rng_state()
    output = evaluate_flat_views(adapter, views)
    assert torch.equal(output[:, :, 0], output[:, :, 4])
    torch.set_rng_state(state)
    adapter(views[:, :, 0])
    expected = torch.get_rng_state()
    assert torch.equal(torch.get_rng_state(), expected)
```

- [ ] **步骤 2：运行测试并确认缺少实现**

运行：`pytest -q tests/test_counterfactual_readout.py`

预期：导入 `gcnet_missing_m3.counterfactual_readout` 失败。

- [ ] **步骤 3：实现共享 Dropout Flat views**

```python
def evaluate_flat_views(adapter, views):
    normalized = adapter[0](views)
    hidden = adapter[2](adapter[1](normalized))
    if adapter.training and adapter[3].p:
        mask = adapter[3](torch.ones_like(hidden[..., :1, :]))
        hidden = hidden * mask
    return adapter[4](hidden)
```

实现必须验证 adapter 精确为 `LayerNorm/Linear/GELU/Dropout/Linear`，仅抽取一次与原 full view 同分布的 mask，并保证 default-off 不调用该函数。

- [ ] **步骤 4：运行共享视图测试**

运行：`pytest -q tests/test_counterfactual_readout.py -k 'dropout or equal_views'`

预期：PASS，且 RNG advance 与一次原 Flat adapter 调用一致。

- [ ] **步骤 5：提交随机性基础**

```bash
git add gcnet_missing_m3/counterfactual_readout.py tests/test_counterfactual_readout.py
git commit -m "Make evidence interventions share the Flat stochastic path"
```

### 任务 2：实现 CEMR 与 CIR

**文件：**
- 修改：`gcnet_missing_m3/counterfactual_readout.py`
- 修改：`tests/test_counterfactual_readout.py`

- [ ] **步骤 1：添加边际与 coalition 失败测试**

```python
def test_marginal_and_pair_effects_match_manual_flat_calls():
    result = block.decompose(local, base, gap, availability, umask,
                             local_skip=skip, adapter=adapter)
    torch.testing.assert_close(result.marginal[0], result.full - result.omit[0])
    torch.testing.assert_close(
        result.pair[0, 1],
        result.value[(0, 1)] - result.value[(0,)]
        - result.value[(1,)] + result.value[()],
    )
```

同时断言 inactive Gap、首句和 padding 的 token/residual 严格为零。

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/test_counterfactual_readout.py -k 'marginal or pair or mask'`

预期：CEMR/CIR 类或 decomposition API 尚不存在。

- [ ] **步骤 3：实现共同 token encoder**

```python
class EvidenceEffectEncoder(nn.Module):
    def __init__(self, token_count, output_dim):
        self.effect = nn.Linear(output_dim, 128)
        self.type = nn.Embedding(token_count, 128)
        self.availability = nn.Linear(3, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.residual = nn.Linear(128, output_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
```

编码前后都 hard-mask inactive token；active-count mean 使用 `clamp_min(1)`，无 active evidence 时 residual 为零。

- [ ] **步骤 4：实现 CEMR 固定五视图**

构造 full 与四个 omit 输入，计算 `d_i = u_full - u_omit_i`。Local skip 只计算一次并加入所有 view；输出 anchor、residual 和 detached diagnostics。

- [ ] **步骤 5：实现 CIR 固定十二视图**

构造 empty、四个 singleton、六个 pair 和 full 输入，计算四个 `I_i` 和六个 `I_ij`。只保留 active singleton/pair；不增加三阶或四阶项。

- [ ] **步骤 6：运行模块测试**

运行：`pytest -q tests/test_counterfactual_readout.py`

预期：全部通过，包括 zero-init、公式、NaN inactive slot、padding、finite gradient 和 diagnostics。

- [ ] **步骤 7：提交读出模块**

```bash
git add gcnet_missing_m3/counterfactual_readout.py tests/test_counterfactual_readout.py
git commit -m "Expose Flat evidence effects without replacing its anchor"
```

### 任务 3：接入 OSRAM 与配置系统

**文件：**
- 修改：`gcnet_missing_m3/osram.py`
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 修改：`tests/test_counterfactual_readout.py`

- [ ] **步骤 1：添加默认兼容与冲突失败测试**

```python
def test_default_state_output_rng_and_memory_match_legacy(): ...

@pytest.mark.parametrize('mode', ['marginal', 'coalition'])
def test_enabled_zero_init_matches_flat_and_preserves_memory(mode): ...

def test_config_rejects_other_readout_blocks():
    with pytest.raises(ValueError):
        TrainConfig(osram_counterfactual_readout='marginal', osram_relation_block=True)
```

- [ ] **步骤 2：运行测试确认配置不存在**

运行：`pytest -q tests/test_counterfactual_readout.py -k 'default or enabled or config'`

预期：新配置字段或 OSRAM 参数不存在。

- [ ] **步骤 3：增加配置与 CLI**

在 `TrainConfig` 增加 `osram_counterfactual_readout: str = 'none'`，CLI choices 为 `none/marginal/coalition`。验证 causal cfg84 Flat、emotion-only、无其他 readout block，并在 config JSON 中记录非默认模式。

- [ ] **步骤 4：接入 Flat readout**

在原 Flat 分支中，`none` 完全沿用旧表达式；启用时调用新模块获得同一个 `flat_anchor` 与 residual：

```python
flat_anchor, residual = self.counterfactual_readout(
    local, emotion_base, emotion_gap, availability, umask,
    local_skip=self.local_skip, adapter=self.emotion_adapter)
hidden = self.emotion_norm(flat_anchor + residual)
```

- [ ] **步骤 5：运行回归测试**

运行：

```bash
pytest -q tests/test_counterfactual_readout.py
pytest -q tests/test_missing_m3.py tests/test_base_first_gap.py tests/test_coalition_mobius.py
```

预期：全部通过；旧路径 state/output/RNG 不变。

- [ ] **步骤 6：提交集成**

```bash
git add gcnet_missing_m3/osram.py gcnet_missing_m3/model.py gcnet_missing_m3/train_gcnet.py tests/test_counterfactual_readout.py
git commit -m "Integrate optional Flat evidence-effect readouts"
```

### 任务 4：建立可恢复的双实验 runner

**文件：**
- 创建：`experiments/osram_counterfactual_readouts_20261011/run.py`
- 创建：`experiments/osram_counterfactual_readouts_20261011/RESULT.md`
- 创建：`tests/test_counterfactual_readout_runner.py`

- [ ] **步骤 1：编写 runner 合同失败测试**

```python
@pytest.mark.parametrize('mode', ['marginal', 'coalition'])
def test_candidate_changes_only_requested_readout(mode):
    result = candidate_config(reference, mode)
    assert delta(reference, result) == {'osram_counterfactual_readout': mode}
```

测试还要拒绝 seed 非66、非 Test-oracle、非 cyclic、Nested/JEPA/Gate/completion，以及非 `100@.001 + 50@.0001` 恢复。

- [ ] **步骤 2：运行 runner 测试确认失败**

运行：`pytest -q tests/test_counterfactual_readout_runner.py`

预期：实验 runner 不存在。

- [ ] **步骤 3：实现 prepare/smoke/train/resume/summarize**

Runner 生成 immutable source snapshot、MANIFEST、数据 hash、独立 A/B 输出、完整训练状态与每率 best checkpoint。第一阶段100 epoch；第二阶段从模型/优化器/RNG状态恢复并将 LR 改为0.0001，完成至150。

- [ ] **步骤 4：运行 runner 单测与静态检查**

运行：

```bash
pytest -q tests/test_counterfactual_readout_runner.py
python -m py_compile experiments/osram_counterfactual_readouts_20261011/run.py
```

预期：PASS。

- [ ] **步骤 5：提交 runner**

```bash
git add experiments/osram_counterfactual_readouts_20261011 tests/test_counterfactual_readout_runner.py
git commit -m "Make A and B readout screens independently recoverable"
```

### 任务 5：远程 correctness 与真实 MOSI smoke

**文件：**
- 修改：`experiments/osram_counterfactual_readouts_20261011/RESULT.md`

- [ ] **步骤 1：同步 committed snapshot 到 biggpu 健康卡**

先运行 `nvidia-smi` 获取 host index/UUID，明确排除 GPU4。把 committed source 复制到新的不可变目录，不覆盖任何历史实验。

- [ ] **步骤 2：运行目标测试**

运行新模块、集成和 runner 测试。预期全部 PASS；同一错误未修复前不重试训练。

- [ ] **步骤 3：分别运行 A/B 真实 MOSI 三步 smoke**

每种模式完成 forward/backward/clip/Adam step，记录 loss、梯度、参数更新、峰值显存、view count 和 wall time。Smoke 不作为性能结果。

- [ ] **步骤 4：验证单卡双并发容量**

以两个短进程错开启动，确认总峰值显存低于32GB并留至少4GB余量，且没有明显 OOM/CPU loader 故障。若不足，则记录证据并采用同卡顺序队列，不改变 batch size。

- [ ] **步骤 5：提交 smoke 证据**

```bash
git add experiments/osram_counterfactual_readouts_20261011/RESULT.md
git commit -m "Verify evidence-effect readouts on real MOSI batches"
```

### 任务 6：启动两个150-epoch正式筛选

**文件：**
- 创建：`experiments/osram_counterfactual_readouts_20261011/LAUNCH.md`

- [ ] **步骤 1：启动 A/B 独立任务**

在同一健康 GPU 上以独立 PID、日志、目录和锁启动。并发容量通过则同时运行，否则由一个持久 dispatcher 在同一 GPU 顺序补位。

- [ ] **步骤 2：验证真实运行状态**

同时核对进程、目标 GPU UUID、显存、日志中的数据维度/epoch、PROVENANCE 状态和输出隔离；仅出现 PID 不算启动成功。

- [ ] **步骤 3：写入并提交启动记录**

记录 commit、source snapshot、服务器、host GPU/UUID、PID、完整命令、输出、日志、恢复协议和完成条件。

```bash
git add experiments/osram_counterfactual_readouts_20261011/LAUNCH.md
git commit -m "Record the paired evidence-readout screening launch"
```

- [ ] **步骤 4：完成后汇总，不自动扩种子**

核对150 history、八个 checkpoints/predictions、mask hash 和 artifact hashes；报告每率 W-F1、mean8、high、selected epoch、显存和时间。结果提交 Git，但不自动 push，也不启动 seed67/68。
