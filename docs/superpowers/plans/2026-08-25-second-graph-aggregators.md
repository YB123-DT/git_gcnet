# GCNet 第二层聚合器实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 在不重新训练 Original 的前提下，分两阶段完成四个可训练图候选：Phase A 为 GenAgg 与 scaled Soft Medoid；Phase B 为 SSMA Conv2 与自定义 RTDR。每阶段运行 IEMOCAPSix fold-5 的12任务判别实验，共24个候选任务。

**范围更新（2026-08-25）：** 下文任务1–8原文是已执行的Phase A计划，保留作为provenance，不回写成“仿佛一开始就包含全部候选”。Phase B按 [新增候选决策](../specs/2026-08-25-additional-graph-candidates-design.md) 执行：SSMA只替换两支conv2聚合；纯线性RTLF因与Original等价被否决；非平凡零参数实验明确命名为RTDR，只改变二跳relation-transition mask。Ego–Neighbor Separation和Centered Clipping只形成否决证据，不注册训练任务。

**架构：** 新聚合器封装在独立文件中，并通过 `GraphNetwork.conv2` 的单一选择器接入。训练循环只为 GenAgg 加入来源要求的 inverse-consistency loss；现有 runner 仅增加候选 arm 和可选并行-arm 模式，以复用全部 manifest、锁、resume 和进程清理逻辑。

**技术栈：** Python 3.8、PyTorch 1.8、PyG 2.0.1、torch-scatter 2.0.8、unittest、IEMOCAPSix。

英文镜像：[2026-08-25-second-graph-aggregators.en.md](2026-08-25-second-graph-aggregators.en.md)。中文镜像：[2026-08-25-second-graph-aggregators.zh.md](2026-08-25-second-graph-aggregators.zh.md)。

---

## 文件结构

- 创建 `gcnet/second_graph_aggregation.py`：GenAgg、Soft Medoid 与 GraphConv 兼容层。
- 修改 `gcnet/model.py`：构造两个第二层选择器并暴露 inverse auxiliary loss。
- 修改 `gcnet/train_gcnet.py`：CLI、训练期 inverse loss、归档身份。
- 创建 `tests/test_second_graph_aggregation.py`：公式、等价、梯度、参数测试。
- 创建 `tests/test_second_graph_aggregation_integration.py`：模型双分支与 RNG 测试。
- 修改 `tests/test_training_protocol.py`：CLI、loss 与归档回归。
- 修改 `experiments/mpfilm_iemocap6/run_locked_ab.py`：复用 runner 增加两条 arm 与并行-arm 开关。
- 修改 `tests/test_mpfilm_runner.py`：12任务与命令差分测试。
- 创建 `experiments/second_graph_aggregation_iemocap6/summarize_gate.py`：继承 Original 并执行锁定判别。
- 创建 `tests/test_second_graph_aggregation_summary.py`：provenance 与门槛测试。
- 创建三份 `EXPERIMENT`/`RESULTS` Markdown：中英双语实验记录。

### 任务1：先锁定聚合器数学行为

- [ ] **步骤1：编写失败测试**

在 `tests/test_second_graph_aggregation.py` 写入 identity sum 特例、三点 Soft Medoid 手算、单邻居、同质邻域、空目标、参数数和有限梯度测试。核心断言：

```python
expected_weights = torch.softmax(-pairwise.sum(-1) / temperature, dim=0)
expected = messages.size(0) * (expected_weights[:, None] * messages).sum(0)
torch.testing.assert_close(actual, expected)
```

- [ ] **步骤2：确认 RED**

运行：

```bash
PYTHONPATH=gcnet:. /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest tests.test_second_graph_aggregation -v
```

预期：因 `second_graph_aggregation` 不存在而失败。

- [ ] **步骤3：实现最小模块**

在 `gcnet/second_graph_aggregation.py` 实现：

```python
class CompatibleMish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GenAggGraphConv(GraphConv):
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        mean = scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        centered = inputs - self.beta * mean.index_select(0, index)
        encoded = self.forward_map(centered.unsqueeze(-1))
        encoded_mean = scatter_mean(encoded, index, dim=0, dim_size=dim_size)
        degree = scatter_add(
            inputs.new_ones((inputs.size(0), 1)), index, dim=0,
            dim_size=dim_size,
        )
        decoded = self.inverse_map(
            encoded_mean * degree.clamp_min(1).unsqueeze(-1).pow(self.alpha)
        ).squeeze(-1)
        return decoded * degree.gt(0).to(decoded.dtype)
```

Soft Medoid 对 `lin_l.weight` 做无 bias message transform，按目标节点打包至 `[N,K,D]`，计算 `[N,K,K]` 距离，聚合后只加一次 `lin_l.bias` 与原 `lin_r(x)`。

- [ ] **步骤4：运行聚合器测试**

预期：全部通过。

- [ ] **步骤5：提交**

使用 Lore commit，记录来源版本、兼容约束、参数数与未运行实验。

### 任务2：接入 GCNet 且保持 add 精确不变

- [ ] **步骤1：编写失败集成测试**

在 `tests/test_second_graph_aggregation_integration.py` 比较 implicit legacy 与 explicit add：CPU RNG、state key、参数张量、forward、input gradient；并检查两候选只替换两个 `conv2`。

- [ ] **步骤2：确认 RED**

运行该测试模块，预期构造函数不接受新参数。

- [ ] **步骤3：修改 `gcnet/model.py`**

新增 `second_graph_aggregation="add"`，构造逻辑：

```python
if second_graph_aggregation == "add":
    self.conv2 = GraphConv(hidden_size, hidden_size)
else:
    self.conv2 = build_second_graph_conv(
        second_graph_aggregation, hidden_size, hidden_size
    )
```

新参数只在保存并恢复 CPU RNG 的上下文中初始化；GraphConv 原有 `lin_l/lin_r` 必须按 legacy 顺序消耗 RNG。`GraphModel.second_graph_auxiliary_loss()` 返回两个分支当前 inverse loss 之和，其他模式返回同设备标量零。

- [ ] **步骤4：运行集成及旧模型测试**

运行新集成测试、`test_model_mpfilm_integration`、`test_bilstm_ablation`，预期全部通过。

- [ ] **步骤5：提交**

Lore commit 必须记录 default add 的精确兼容证据。

### 任务3：训练、CLI和归档身份

- [ ] **步骤1：编写失败协议测试**

在 `tests/test_training_protocol.py` 验证 choices、legacy default、candidate filename tag、训练期 inverse loss 和 evaluation 零 auxiliary loss。

- [ ] **步骤2：确认 RED**

运行 `tests.test_training_protocol`，预期新 CLI 不存在。

- [ ] **步骤3：修改训练路径**

增加：

```python
parser.add_argument(
    "--second-graph-aggregation",
    choices=("add", "genagg", "soft_medoid"),
    default="add",
)
```

仅训练时：

```python
loss_aggregation = model.second_graph_auxiliary_loss()
if train:
    loss = loss + loss_aggregation
```

候选归档名与 suffix 加入 aggregation tag；`add` 保持旧字符串逐字不变。保存 `second_graph_aggregation` 与 aggregation auxiliary loss 日志，不改变返回 tuple。

- [ ] **步骤4：运行协议回归**

预期 `test_training_protocol`、MPFiLM、Sequence-AFF 与 BiLSTM runner 测试全部通过。

- [ ] **步骤5：提交**

记录训练损失只作用于 GenAgg training view。

### 任务4：扩展既有 runner，禁止 Original 重跑

- [ ] **步骤1：编写失败 runner 测试**

加入 `genagg`、`soft_medoid` 映射，验证12个唯一任务、候选命令固定 `--graph-conv-variant original`、`--branch-fusion addition`，并仅增加相应 `--second-graph-aggregation`。

- [ ] **步骤2：确认 RED**

运行 `tests.test_mpfilm_runner`，预期未知 arm。

- [ ] **步骤3：最小修改 runner**

扩展映射并加入 `--parallel-arms`。默认行为保持逐 arm；指定开关时调用一次：

```python
run_jobs(jobs, gpus, workers_per_gpu, python, repository, data_root, mask_bank_root)
```

正式命令固定：

```text
--stage formal --arms genagg soft_medoid
--rates 0.0 0.7 --seeds 66 67 68
--gpus 0 1 2 3 --workers-per-gpu 3 --parallel-arms
```

runner 不包含 `original` arm，因此不会产生 Original 进程。禁止使用 `stage=gate`，因为它会创建不同路径和不同不可变command identity，不能被后续formal继承。

- [ ] **步骤4：运行 runner 回归**

预期新旧 runner 测试全部通过。

- [ ] **步骤5：提交**

记录复用 runner 的原因与禁止 Original 的 directive。

### 任务5：实现继承式配对汇总

- [ ] **步骤1：编写失败 summary 测试**

创建 `tests/test_second_graph_aggregation_summary.py`，构造通过与逐条件失败 fixture，验证 Original 不能来自候选目录、mask SHA 必须一致、原始精度决定阈值。

- [ ] **步骤2：确认 RED**

预期 summary 模块不存在。

- [ ] **步骤3：实现 `summarize_gate.py`**

复用 CP-LECC archive metrics 与原子写入逻辑，按 `(rate, seed, fold)` 连接已有 Original 与候选。Original 使用专用历史验证器：允许当前新增字段缺失并映射为 legacy 默认值，但严格检查历史run manifest、命令、fold、特征、seed、rate、参数数和mask SHA；候选继续使用当前严格payload验证。输出 task rows、rate means、seed macros及命名判定条件。

- [ ] **步骤4：运行 summary 测试**

预期全部通过。

- [ ] **步骤5：提交**

记录门槛预注册且不得在看到结果后修改。

### 任务6：完整代码验证与远端一次性兼容检查

- [ ] **步骤1：运行完整本地测试**

```bash
PYTHONPATH=gcnet:. /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest discover -s tests -v
```

预期零失败、仅既有真实归档缺失测试 skip。

- [ ] **步骤2：静态检查**

运行 `compileall`、`git diff --check`、Python 3.8 AST parse；预期零错误。

- [ ] **步骤3：建立真实远端 Git 副本**

从干净本地 commit 创建 `git bundle`，传到 biggpu 并 clone 到新目录，避免复用先前无 `.git` 的 staging 副本。比较本地/远端 HEAD 与 source SHA256 manifest。

- [ ] **步骤4：官方训练环境一次性检查**

每个候选只运行一个合成 FP32 forward/backward，不执行 epoch。预期输出有限，GenAgg inverse 参数和 Soft Medoid input/linear 参数梯度有限。

- [ ] **步骤5：修复后重跑唯一完整验证**

若检查失败，按根因修复；不创建新 smoke 脚本，不重复无关测试。

### 任务7：启动并完成Phase A的12任务判别实验

- [ ] **步骤1：验证远端数据和 inherited Original**

确认数据特征目录、固定 mask bank、Original 40 NPZ、命令 provenance 和 mask SHA；只读验证，不训练。

- [ ] **步骤2：启动一次 runner**

四张卡、每卡三个进程，一次启动12个候选任务。记录启动 UTC、PID、GPU、source HEAD 与 manifest。

- [ ] **步骤3：持续监控到终态**

检查每个任务100 epoch、Finish、return code 0、一个 NPZ；异常任务只在根因修复后按同一 task key resume，不启动 Original。

- [ ] **步骤4：同步结果回本地**

回传完整 candidate artifacts、run manifest、invocation、logs和状态；比较 source manifest。

- [ ] **步骤5：运行锁定汇总**

生成主、中文、英文 `RESULTS`，逐候选给出 PASS/FAIL 和每个 rate/seed 的配对证据。

### 任务8：按门槛自动闭环

- [ ] **步骤1：失败候选记录并停止**

写明失败条件、效应方向和适用限制，不继续补 formal。

- [ ] **步骤2：通过候选补齐正式网格**

仅为 PASS 候选补缺少的 rate/seed 任务；首批6个候选任务原样继承，不重复。

- [ ] **步骤3：最终验证与评审**

运行完整测试、结果完整性检查、architect review、changed-files-only deslop及再次回归。

- [ ] **步骤4：提交结果和代码**

使用 Lore commits，包含 Tested、Not-tested、Directive 与真实结果边界；未完成候选不进入完成版本目录。

### 任务9：完成Phase B并自动发布

- [ ] **步骤1：SSMA与RTDR按TDD实现并两阶段审查**

SSMA通过独立tiny oracle、双分支参数增量、Torch1.8 GPU反向；RTDR通过Original原路径bit-exact和full-transition容差等价，再开放diagonal selector。

- [ ] **步骤2：扩展CLI、归档身份与runner**

Phase B固定为 `ssma rtdr` × missing `{0.0,0.7}` × seeds `{66,67,68}`，`stage=formal`，四卡每卡三个进程。不得启动Original。

- [ ] **步骤3：运行、监控、回传和汇总**

两阶段合计24个候选任务均要求return code 0、100 epoch记录、一个NPZ、mask/provenance一致；失败任务只按原task key修复后resume。

- [ ] **步骤4：自动推送指定仓库**

将完成代码、实验说明、manifest、任务级结果和双语汇总同步回 `/data2/yb/paper`，整理为完成版本目录，使用Lore commit提交，并推送到 `https://github.com/YB123-DT/git_gcnet`。失败/未完成任务放入明确的未完成记录或排除，不得混入完成结果。
