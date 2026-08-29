# MOSI Shared Post-Graph BiLSTM 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加一个只共享 Temporal/Speaker 图后 BiLSTM 的可复现实验开关，并完成三种子 validation-only 单变量筛选。

**架构：** 两套 RGCN/GraphConv 和 branch-specific linear 保持独立；Speaker 图分支在候选模式下复用 Temporal `grufusion`。原 Speaker `grufusion` 保留 checkpoint keys 但冻结，从而保持构造 RNG 与 strict loading，同时减少 2,508,000 个可训练参数。

**技术栈：** Python、PyTorch、PyTorch Geometric、pytest、biggpu V100、JSON 实验清单。

---

### 任务 1：锁定模型边界

**文件：**

- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_modality_jepa/model.py`
- 修改：`gcnet_missing_m3/model.py`

- [ ] **步骤 1：编写默认兼容与共享行为红灯测试**

测试必须断言默认/显式 independent 精确等价、候选只复用 Temporal
`grufusion`、两套 graph conv/linear 保持独立、Speaker `grufusion` 冻结且不影响
输出。

- [ ] **步骤 2：运行聚焦测试并确认因缺少开关而失败**

运行：

```bash
scripts/remote_missing_m3.sh test -q tests/test_missing_m3.py -k postgraph_sequence
```

预期：测试因 `postgraph_sequence_mode` 尚未实现而失败。

- [ ] **步骤 3：实现最小模型开关**

在 `GraphNetwork.forward()` 中加入非注册的 recurrent override；在
`GraphModel.encode_hidden()` 的 Speaker 分支按模式传入 Temporal `grufusion`；在
构造结束后冻结保留的 Speaker `grufusion`。

- [ ] **步骤 4：运行聚焦测试并确认通过**

运行同一步骤 2 的命令，预期全部 PASS。

### 任务 2：锁定梯度、参数和 checkpoint

**文件：**

- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] **步骤 1：编写梯度与参数红灯测试**

覆盖 Temporal 共享 BiLSTM 双路梯度、Speaker BiLSTM `grad=None`、可训练参数减少
2,508,000、state keys 不变、strict loading，以及 CPU FP32 finite backward。

- [ ] **步骤 2：添加 TrainConfig 与 CLI 传递**

新增 `--postgraph-sequence-mode {independent,shared-bilstm}`，默认
`independent`。在 config、checkpoint 和 metrics 中记录该值。

- [ ] **步骤 3：运行完整模型测试**

```bash
scripts/remote_missing_m3.sh test -q tests/test_missing_m3.py
```

预期：全部 PASS。

### 任务 3：锁定 runner 与对照

**文件：**

- 修改：`scripts/run_mosi_conditioned_readout.py`
- 修改：`tests/test_mosi_conditioned_readout_runner.py`

- [ ] **步骤 1：编写 runner 红灯测试**

断言独立目录 `shared-postgraph-bilstm/seed_*`、唯一 treatment command、resume
字段、manifest 字段，以及与任何 readout/JEPA/Packed/SmoothL1 组合时拒绝。

- [ ] **步骤 2：强制 direct deterministic control**

候选若未指向包含 `seed_*/config.json` 的 deterministic Legacy 根目录，则在训练
前失败；禁止回退旧 hidden/window controls。

- [ ] **步骤 3：实现最小 runner 传递并运行测试**

```bash
scripts/remote_missing_m3.sh test -q tests/test_mosi_conditioned_readout_runner.py
```

预期：全部 PASS。

### 任务 4：完整验证与远程筛选

**文件：**

- 更新：`experiments/missing_m3_mosi_conditioned_readout_20260829/EXPERIMENT.md`
- 生成：`experiments/missing_m3_mosi_conditioned_readout_20260829/results/shared-postgraph-bilstm/`

- [ ] **步骤 1：运行完整回归、diff-check 和编译检查**

```bash
scripts/remote_missing_m3.sh test -q \
  tests/test_missing_m3.py \
  tests/test_mosi_conditioned_readout.py \
  tests/test_mosi_conditioned_readout_runner.py
git diff --check
python -m py_compile gcnet_modality_jepa/model.py gcnet_missing_m3/model.py \
  gcnet_missing_m3/train_gcnet.py scripts/run_mosi_conditioned_readout.py
```

预期：测试、diff-check 和编译全部通过。

- [ ] **步骤 2：运行短 CUDA FP32 forward/backward**

使用 GPU7 或当时空闲且非 GPU4 的卡；确认参数数量、共享梯度和 finite 输出。

- [ ] **步骤 3：启动三种子 validation-only screen**

三个 seeds 并发运行；显式传入 deterministic Legacy direct control，且不启动新
control。

- [ ] **步骤 4：同步并审计结果**

核验 100 epochs、无 test NPZ、三组 config audit、source/control SHA、gate 和逐
seed/per-rate delta。失败则关闭路线；通过才扩 seeds 69、70。

### 任务 5：提交与同步

**文件：**

- 提交：上述源码、测试、规格、计划和完整结果记录。

- [ ] **步骤 1：运行完成前审查与回归复验**

执行架构审查、changed-files deslop 和最终测试。

- [ ] **步骤 2：使用 Lore 协议提交**

提交信息必须记录唯一变量、被拒绝的组合方案、验证命令和未运行的 test。

- [ ] **步骤 3：推送 GitHub 分支**

推送到 `github/feature/m3-jepa-gcnet`，只在候选结果完整后进行。

