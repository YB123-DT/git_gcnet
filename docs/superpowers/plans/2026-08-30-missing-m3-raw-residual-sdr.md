# Missing-M3 Raw-Residual SDR 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> superpowers-zh:subagent-driven-development（推荐）或
> superpowers-zh:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）
> 跟踪进度。

**目标：** 让 SDR-public 直接接收保留 A/T/V 原始宽度的 2560D Raw-Residual
输入，在不改变 Missing-M3 predictor、损失或 mixed-rate 协议的条件下完成 MOSI
五种子诊断。

**架构：** 复用现有 `RawResidualObservedEncoder` 产生 2560D 模态块和 256D
Student latents；共享 SDR 模型增加向后兼容的 node-input switch，默认 Slot 路径不变。
新目录把该 switch 锁死为 Raw-Residual + SDR-public，并用一个 checkpoint 测试八个
missing rates。

**技术栈：** Python 3.8/3.10、PyTorch、PyTorch Geometric、pytest、CMU-MOSI
冻结 wav2vec/DeBERTa/MANet 特征、biggpu V100、Git/GitHub。

---

## 文件结构

**修改：**

- `gcnet_missing_m3_sdr_backbone/model.py`：增加默认不变的 SDR node-input switch。
- `gcnet_missing_m3_sdr_backbone/train_gcnet.py`：增加可注入 model builder/result
  identity 的训练生命周期复用点，默认行为不变。
- `gcnet_missing_m3_sdr_backbone/tests/test_integration.py`：锁定 Slot 回归兼容和
  Raw-Residual 输入不变量。
- `gcnet_missing_m3_sdr_backbone/tests/test_train_gcnet.py`：锁定默认 trainer 输出不变。

**创建：**

- `gcnet_missing_m3_raw_sdr/__init__.py`：只公开正式 Raw-SDR 类型。
- `gcnet_missing_m3_raw_sdr/model.py`：锁定 Raw-Residual + SDR-public 的薄模型。
- `gcnet_missing_m3_raw_sdr/train_gcnet.py`：五种子正式训练入口。
- `gcnet_missing_m3_raw_sdr/run_mosi.py`：只调度五个 treatment job，继承控制结果。
- `gcnet_missing_m3_raw_sdr/README.md`：架构、命令与边界。
- `gcnet_missing_m3_raw_sdr/STATUS.md`：运行状态与结论。
- `gcnet_missing_m3_raw_sdr/tests/test_model.py`：模型、mask、梯度测试。
- `gcnet_missing_m3_raw_sdr/tests/test_train_gcnet.py`：配置和结果身份测试。
- `gcnet_missing_m3_raw_sdr/tests/test_runner.py`：5-job、GPU、resume、汇总测试。
- `gcnet_missing_m3_raw_sdr/results/`：compact summary 与 provenance，不保存大 checkpoint。

## 任务 1：TDD 增加向后兼容的 SDR 输入开关

**文件：**

- 修改：`gcnet_missing_m3_sdr_backbone/tests/test_integration.py`
- 修改：`gcnet_missing_m3_sdr_backbone/model.py`

- [ ] **步骤 1：编写默认 Slot 回归测试**

在生产代码改动前，构造同 seed 的旧调用与显式 `sdr_input_type="slot"` 调用：

```python
torch.manual_seed(7)
implicit = _candidate("sdr-public")
implicit_state = implicit.state_dict()
torch.manual_seed(7)
explicit = _candidate("sdr-public", sdr_input_type="slot")
assert implicit_state.keys() == explicit.state_dict().keys()
for key, value in implicit_state.items():
    torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
assert explicit.conversation_backbone.input_dim == 256
```

- [ ] **步骤 2：编写 Raw-Residual 红灯测试**

期望调用：

```python
model = _candidate(
    "sdr-public",
    fusion_type="raw-residual",
    sdr_input_type="raw-residual",
)
assert isinstance(model.observed_set, RawResidualObservedEncoder)
assert model.conversation_backbone.input_dim == 512 + 1024 + 1024
```

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdr_backbone/tests/test_integration.py \
  -k "sdr_input or raw_residual" -q
```

预期：Raw-Residual 测试因未知 `sdr_input_type` 或 Slot lock 失败；默认 Slot 回归通过。

- [ ] **步骤 3：实现最小 switch**

`MissingM3SDRModel.__init__` 新增 keyword-only：

```python
sdr_input_type="slot"
```

只接受 `slot` 与 `raw-residual`；二者分别要求同名 `fusion_type`，SDR input width
分别为 `latent_dim` 与 `sum(dimensions)`。删除继承主干时使用 `hasattr`，因为
Raw-Residual 父类不会注册前图 LSTM/GRU。默认 Slot 路径不增加参数且不改变初始化
顺序。

- [ ] **步骤 4：验证绿灯与完整 SDR 回归**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdr_backbone/tests -q
```

预期：全部通过；现有 verified parameter counts 不变。

- [ ] **步骤 5：提交共享 seam**

Lore commit 必须记录默认 Slot state/RNG 等价与 Raw 输入宽度测试。

## 任务 2：TDD 建立独立 Raw-SDR 模型与训练入口

**文件：**

- 创建：`gcnet_missing_m3_raw_sdr/__init__.py`
- 创建：`gcnet_missing_m3_raw_sdr/model.py`
- 创建：`gcnet_missing_m3_raw_sdr/train_gcnet.py`
- 创建：`gcnet_missing_m3_raw_sdr/tests/__init__.py`
- 创建：`gcnet_missing_m3_raw_sdr/tests/test_model.py`
- 创建：`gcnet_missing_m3_raw_sdr/tests/test_train_gcnet.py`
- 修改：`gcnet_missing_m3_sdr_backbone/train_gcnet.py`
- 修改：`gcnet_missing_m3_sdr_backbone/tests/test_train_gcnet.py`

- [ ] **步骤 1：写模型红灯测试**

测试正式类只能构造成：

```python
model = MissingM3RawSDRModel(...)
assert model.sdr_variant == "sdr-public"
assert model.sdr_input_type == "raw-residual"
assert model.conversation_backbone.input_dim == 2560
```

传入 `fusion_type="slot"`、`sdr_variant="sdr-paper"` 或任何第二分支配置必须抛出
`ValueError`。

- [ ] **步骤 2：写七 pattern、泄漏和零初始化测试**

对 A/T/V/AT/AV/TV/ATV 逐一断言：

```python
encoded, latents = model.observed_set(features, availability, umask)
expected = features * expand_availability(availability, dimensions)
torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
assert torch.count_nonzero(encoded[missing_or_padding]) == 0
```

复制 batch 后只改变 missing blocks，`encoded`、observed latents 和
`predict_missing=False` logits 必须不变。

- [ ] **步骤 3：写 backward 红灯测试**

执行 natural classification + JEPA forward/backward，检查 finite nonzero gradient
到达：Student projector、residual adapter、SDR pre-GRU、temporal graph branch、
regression head 与 Missing-M3 predictor。Teacher 参数保持无梯度。

- [ ] **步骤 4：实现薄模型**

`MissingM3RawSDRModel` 复用共享类，只注入：

```python
fusion_type="raw-residual"
representation_type="slot"
sdr_input_type="raw-residual"
sdr_variant="sdr-public"
```

不得复制 `SDRConversationBackbone` 或 `RawResidualObservedEncoder`。

- [ ] **步骤 5：先写 trainer hook 红灯测试**

共享 `run_experiment` 接受 keyword-only `model_builder` 与 `result_identity`；不传时
旧 trainer 的 config、metrics identity 和参数量保持原值。新 trainer 调用 hook 后：

```python
metrics["variant"] == "raw-residual-sdr-public"
metrics["sdr_variant"] == "sdr-public"
metrics["sdr_input_type"] == "raw-residual"
metrics["backbone"] == "raw-residual-sdr-public"
```

- [ ] **步骤 6：实现最小 trainer 复用点与锁定配置**

`RawSDRTrainConfig` 仅开放 `seed/device/epochs/evaluate_test`；其余值全部锁定，特别是
`fusion_type="raw-residual"`、`sdr_variant="sdr-public"`、lr `5e-4`、100 epoch
和 `train_rate_mode="all"`。新 CLI 不暴露结构搜索参数。

- [ ] **步骤 7：验证绿灯并提交**

运行新模型/trainer 测试和旧 SDR 完整测试，确认旧路径无回归后提交。

## 任务 3：TDD 实现只包含五个 treatment 的 runner

**文件：**

- 创建：`gcnet_missing_m3_raw_sdr/run_mosi.py`
- 创建：`gcnet_missing_m3_raw_sdr/tests/test_runner.py`

- [ ] **步骤 1：写 5-job 与 GPU 红灯测试**

```python
jobs = build_jobs(output_root=tmp_path)
assert [job.seed for job in jobs] == [66, 67, 68, 69, 70]
assert [job.gpu for job in jobs] == [2, 3, 5, 6, 7]
assert len(jobs) == 5
assert all(job.variant == "raw-residual-sdr-public" for job in jobs)
```

任何 GPU 4、重复 GPU、未知 seed 或额外 variant 均必须被拒绝。命令必须包含
`--train-rate-mode all`、`--lr 0.0005`，且不能出现 Original、Slot 或
`sdr-paper` 训练项。

- [ ] **步骤 2：写结果完整性红灯测试**

一个 job 只有在以下条件同时满足时才可 resume 为 complete：100 epoch history、
8-rate validation/test、40 prediction fields 所需 NPZ、正确 config、参数量、
schedule mask SHA、prediction availability SHA 和 producer provenance 均有效。
半写文件、非零退出但完整结果、旧目录重标记均按现有 SDR runner 语义处理。

- [ ] **步骤 3：写继承对照与 validation-only gate 测试**

汇总读取现有 GCNet Control 和 Slot-SDR-public compact metrics，不产生它们的训练
command。Primary gate 只比较 validation；交换 test 分数不得改变 gate/status。

- [ ] **步骤 4：实现 runner**

优先复用 `gcnet_missing_m3_sdr_backbone.run_mosi` 的审计函数；新文件只维护 treatment
identity、五个 job、命令和两组 inherited reference。每张健康卡只启动一个 job，
异常进程超时终止，不创建 1-epoch 保存兼容任务。

- [ ] **步骤 5：运行 runner 测试并提交**

运行新 runner 测试、旧 runner 测试与 `py_compile`；记录 exact pass count。

## 任务 4：一次 smoke 后直接运行五种子正式实验

**文件：**

- 创建：`gcnet_missing_m3_raw_sdr/README.md`
- 创建：`gcnet_missing_m3_raw_sdr/STATUS.md`

- [ ] **步骤 1：同步到 biggpu**

只同步当前 worktree 的 tracked source；使用已记录的：

```text
Python: /data2/yb/reproduction_envs/gcnet-official/bin/python
Features: /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features
GPUs: 2,3,5,6,7
```

不重复扫描 Python 环境，不使用坏卡 GPU 4。

- [ ] **步骤 2：运行唯一 smoke**

只运行 seed 66、1 epoch、`evaluate_test=False`，验证真实 MOSI batch 的 forward、
backward、EMA、八 rate validation、显存和结果写入。若通过，不再运行第二个 smoke。

- [ ] **步骤 3：记录正式参数量并锁定测试**

从 smoke 的模型实例读取 registered/trainable total 与 backbone 参数量，将 exact
整数写入新 runner 的 completion check，再重跑 runner 单测。

- [ ] **步骤 4：启动五个正式 job**

GPU 2/3/5/6/7 各一个 seed，100 epoch。保留每 epoch validation 轨迹；完成后删除
大 checkpoint，只保存 checkpoint SHA256、metrics/history、prediction NPZ、日志与
producer provenance。

- [ ] **步骤 5：监控而不改变模型**

每次状态检查只读取进程、epoch、错误和显存；不在队列运行期间新增模块、修改 loss、
更换 mask 或启动额外 seed。异常 job 只按同 config/provenance resume。

## 任务 5：分析、归档、验证并推送

**文件：**

- 修改：`gcnet_missing_m3_raw_sdr/STATUS.md`
- 创建：`gcnet_missing_m3_raw_sdr/results/formal/summary.json`
- 创建：`gcnet_missing_m3_raw_sdr/results/formal/manifest.json`
- 创建：`gcnet_missing_m3_raw_sdr/results/formal/RESULTS.md`

- [ ] **步骤 1：独立重算 40 个 W-F1**

从 prediction NPZ 使用 MOSI nonzero-label weighted-F1 规则重算，与 metrics.json 比较；
最大绝对误差必须小于 `1e-10`。检查 40/40 schedule hashes 与配对 Control 一致。

- [ ] **步骤 2：只用 validation 做结论**

报告五种子逐 seed、逐 rate、八 rate mean 与 high-missing mean；分别对比
Slot-SDR-public 和 GCNet Control。test 只在 validation 结论之后描述。

- [ ] **步骤 3：更新状态文档**

明确写 PASS/FAIL、支持或否定的假设、参数/时间/显存、padding-safe adaptation 边界
和剩余风险。失败结果也必须完整保留，防止未来重复探索。

- [ ] **步骤 4：完成前验证**

运行：

```bash
/data2/yb/reproduction_envs/s0/bin/python -m pytest \
  gcnet_missing_m3_sdr_backbone/tests \
  gcnet_missing_m3_raw_sdr/tests -q
/data2/yb/reproduction_envs/s0/bin/python -m py_compile \
  gcnet_missing_m3_raw_sdr/*.py
git diff --check
```

核对没有待运行 job、没有已知错误、结果单元为 40/40。

- [ ] **步骤 5：Lore commit 与推送**

提交只包含源代码、测试、文档和 compact results；不提交 checkpoint、完整 feature、
缓存或临时日志。推送当前研究分支到 `https://github.com/YB123-DT/git_gcnet`，并记录
commit SHA 与远程分支。
