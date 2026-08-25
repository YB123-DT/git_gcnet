# GCNet BiLSTM 2×2 因子消融实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现图前/图后 BiLSTM 的四格因子消融，并在 GCNet 四个数据集、八档缺失率下生成可配对、可审计的结果。

**架构：** 每个 BiLSTM 位置同时实例化 recurrent path 与 utterance-wise Linear adapter，用两个 context mode 选择实际前向路径；adapter 初始化不改变官方参数 RNG。新增 dataset-registry runner 隔离数据集 mask 命名空间并统一调度四格，新增汇总器按数据集计算主效应、交互项和坍塌诊断。

**技术栈：** Python 3.8 训练环境、PyTorch 1.8、PyG 2.0.1、NumPy、scikit-learn、SciPy、unittest、V100 CUDA。

---

## 文件职责

- 修改 `gcnet/model.py`：实现共享模块的 pre/post context 选择和 selected-path 参数计数。
- 修改 `gcnet/train_gcnet.py`：增加两个 context CLI、传递设置并保存 provenance。
- 创建 `experiments/bilstm_ablation/run_factorial.py`：四数据集 registry、任务网格、GPU 调度和不可变 artifact 校验。
- 创建 `experiments/bilstm_ablation/summarize.py`：可信归档读取、配对审计、因子效应和 collapse 汇总。
- 创建 `experiments/bilstm_ablation/EXPERIMENT.zh.md`：协议、运行状态、结果和结论。
- 创建 `tests/test_bilstm_ablation.py`：模型结构、shape、梯度、初始化与参数计数测试。
- 创建 `tests/test_bilstm_runner.py`：四数据集任务网格、命令和 provenance 测试。
- 创建 `tests/test_bilstm_summary.py`：合成归档的配对统计和错误路径测试。

### 任务 1：锁定四格模型契约

**文件：**
- 创建：`tests/test_bilstm_ablation.py`
- 修改：`gcnet/model.py`

- [ ] **步骤 1：编写失败的四格构造测试**

测试以 `pre_graph_context`/`post_graph_context` 的 `bilstm|linear` 笛卡尔积构造 `GraphModel`，断言三个位置同时拥有 BiLSTM 和 Linear，但 `forward` 选择的 mode 正确；两个 post 分支必须同步。

```python
for pre in ("bilstm", "linear"):
    for post in ("bilstm", "linear"):
        model = make_model(pre, post)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.pre_graph_projection, nn.Linear)
        assert model.pre_graph_context == pre
        for branch in (model.graph_net_temporal, model.graph_net_speaker):
            assert isinstance(branch.grufusion, nn.LSTM)
            assert isinstance(branch.post_graph_projection, nn.Linear)
            assert branch.post_graph_context == post
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=gcnet python -m unittest tests.test_bilstm_ablation -v
```

预期：因新构造参数或 projection 属性不存在而 FAIL。

- [ ] **步骤 3：实现共享模块和前向选择**

在 `GraphModel.__init__` 中先构造官方 `self.lstm`，再从 forked RNG 初始化：

```python
state = torch.get_rng_state()
self.pre_graph_projection = nn.Linear(adim + tdim + vdim, 2 * D_e)
torch.set_rng_state(state)
```

在 `GraphNetwork.__init__` 中以相同方式创建
`post_graph_projection = nn.Linear(D_h, 2 * D_h)`。前向只选择一个路径：

```python
if self.pre_graph_context == "bilstm":
    outputs, _ = self.lstm(inputfeats[0])
else:
    outputs = self.pre_graph_projection(inputfeats[0])
```

```python
if self.post_graph_context == "bilstm":
    outputs = self.grufusion(outputs)[0]
else:
    outputs = self.post_graph_projection(outputs)
```

保持后续 MatchingAttention/Linear/ReLU 不变。

- [ ] **步骤 4：加入 shape、局部性与梯度测试**

四格均断言 logits/reconstruction/hidden shape 相同且有限。对 Linear path 单独输入两个只在另一 timestep 不同的张量，断言目标 timestep 输出不变。反向后断言被选择模块有梯度、被 bypass 模块梯度为 `None`。

- [ ] **步骤 5：加入 RNG 与官方 on/on 等价测试**

相同 seed 分别构造旧签名默认模型和显式 `bilstm/bilstm`，断言全部官方参数逐位一致、构造后 CPU RNG state 一致；eval 模式同输入输出逐位一致。

- [ ] **步骤 6：实现 selected-path 参数计数并测试**

新增 `selected_path_parameter_count()`，从总参数中排除未选择的 factor counterpart；锁定 Original graph 的四格计数：

```python
EXPECTED = {
    ("bilstm", "bilstm"): 34_140_166,
    ("linear", "bilstm"): 29_782_166,
    ("bilstm", "linear"): 15_110_166,
    ("linear", "linear"): 10_752_166,
}
```

- [ ] **步骤 7：运行聚焦测试并提交**

运行：

```bash
PYTHONPATH=gcnet python -m unittest tests.test_bilstm_ablation tests.test_model_mpfilm_integration -v
```

预期：全部 PASS。按 Lore 协议提交模型与测试。

### 任务 2：CLI 与训练归档 provenance

**文件：**
- 修改：`gcnet/train_gcnet.py`
- 修改：`tests/test_training_protocol.py`
- 修改：`tests/test_bilstm_ablation.py`

- [ ] **步骤 1：编写失败的 CLI 测试**

CLI 必须接受且默认：

```text
--pre-graph-context {bilstm,linear}   default=bilstm
--post-graph-context {bilstm,linear}  default=bilstm
```

使用 Python 3.8 兼容的 `choices`，不使用 `BooleanOptionalAction`。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=gcnet python -m unittest tests.test_training_protocol -v
```

预期：parser 不认识 context 参数而 FAIL。

- [ ] **步骤 3：传递参数并保存 provenance**

`build_model` 将两个 mode 传给 `GraphModel`。结果文件名追加：

```python
f"_prectx:{args.pre_graph_context}_postctx:{args.post_graph_context}"
```

NPZ 新增：

```python
selected_path_parameter_count=np.array(model.selected_path_parameter_count())
```

保留现有 `parameter_count` 作为 stored total。

- [ ] **步骤 4：smoke archive 测试**

现有 short-run fixture 运行 Linear/Linear，加载可信 NPZ，断言 args、文件名、stored total 和 selected-path count 均存在且匹配。

- [ ] **步骤 5：运行测试并提交**

```bash
PYTHONPATH=gcnet python -m unittest tests.test_training_protocol tests.test_bilstm_ablation -v
```

预期：全部 PASS。按 Lore 协议提交。

### 任务 3：四数据集通用任务调度器

**文件：**
- 创建：`experiments/bilstm_ablation/__init__.py`
- 创建：`experiments/bilstm_ablation/run_factorial.py`
- 创建：`tests/test_bilstm_runner.py`

- [ ] **步骤 1：编写 dataset registry 与网格失败测试**

registry 锁定：

```python
DATASETS = {
    "IEMOCAPFour": {"directory": "IEMOCAP", "fold": 5, "metric": "multiclass_weighted_f1"},
    "IEMOCAPSix": {"directory": "IEMOCAP", "fold": 5, "metric": "multiclass_weighted_f1"},
    "CMUMOSI": {"directory": "CMUMOSI", "fold": None, "metric": "nonzero_binary_weighted_f1"},
    "CMUMOSEI": {"directory": "CMUMOSEI", "fold": None, "metric": "nonzero_binary_weighted_f1"},
}
ARMS = {
    "original": ("bilstm", "bilstm"),
    "no_pre_bilstm": ("linear", "bilstm"),
    "no_post_bilstm": ("bilstm", "linear"),
    "no_all_bilstm": ("linear", "linear"),
}
```

断言 formal grid 为 640 jobs；smoke 为 16 jobs；pilot 为 96 jobs。

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTHONPATH=. python -m unittest tests.test_bilstm_runner -v
```

预期：runner 模块不存在而 FAIL。

- [ ] **步骤 3：实现 Python 3.8 兼容 runner**

复用 `run_locked_ab.py` 的锁、日志、status、单归档和 GPU worker 逻辑，但不得导入其 Python 3.10 `| None` 注解。命令按 registry：

- 显式 `--data-root <dataset_root>/<directory>`；
- IEMOCAP 传 `--fold-index 5`，CMU 不传；
- `--mask-bank-root <mask_root>/<dataset>/<split_tag>`；
- 固定 feature names、100 epochs、hidden=200、window=2、lr=0.001、dropout=0.5、batch=32；
- arm 映射为两个 context choices；
- 输出路径包含 dataset/arm/rate/seed/split。

- [ ] **步骤 4：实现不可变 provenance**

run manifest 记录 git HEAD/clean、解释器实际版本、数据 label/feature fingerprints、split tag、metric、训练参数和 GPU。每个 command/status/log/saved 必须完整；partial、stale lock、命令漂移均拒绝续跑。

- [ ] **步骤 5：测试配对与数据集隔离**

断言同 dataset/rate/seed 的四格命令使用同一 mask namespace；不同 dataset 或 split 绝不共享 bank 文件路径。断言 CMU 命令无 fold5。

- [ ] **步骤 6：运行测试并提交**

```bash
PYTHONPATH=. python -m unittest tests.test_bilstm_runner -v
```

预期：全部 PASS。按 Lore 协议提交。

### 任务 4：可信汇总器与因子统计

**文件：**
- 创建：`experiments/bilstm_ablation/summarize.py`
- 创建：`tests/test_bilstm_summary.py`

- [ ] **步骤 1：编写合成归档失败测试**

为四格写入合成 trusted NPZ，断言按 dataset/rate/seed 对齐；任一 mask hash、split hash、label hash、metric、seed、rate、context mode、100 epochs 或 selected count 不匹配均拒绝。

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTHONPATH=. python -m unittest tests.test_bilstm_summary -v
```

预期：summary 模块不存在而 FAIL。

- [ ] **步骤 3：实现指标与因子效应**

ERC 从保存 logits 计算 multiclass weighted-F1/accuracy；MOSI/MOSEI 去掉 label==0 后按符号计算 binary weighted-F1/accuracy。每 dataset/rate 输出 mean、sample SD、paired deltas、win count、paired t、Wilcoxon 和 collapse coverage。

每 dataset/seed 先取八档宏平均，再计算：

```python
pre_effect = y_original - y_no_pre
post_effect = y_original - y_no_post
interaction = y_original - y_no_pre - y_no_post + y_no_all
```

- [ ] **步骤 4：实现机器可读与中文表格输出**

同时写 `summary.json` 和 `SUMMARY.zh.md`，禁止跨数据集 raw metric pooling；IEMOCAP 标注 fold-5 screening 和 validation/test 样本重用限制。

- [ ] **步骤 5：运行测试并提交**

```bash
PYTHONPATH=. python -m unittest tests.test_bilstm_summary -v
```

预期：全部 PASS。按 Lore 协议提交。

### 任务 5：全套本地验证与远程环境门

**文件：**
- 修改：`experiments/bilstm_ablation/EXPERIMENT.zh.md`

- [ ] **步骤 1：运行全套测试**

```bash
PYTHONPATH=gcnet:. python -m unittest discover -s tests -v
python -m py_compile gcnet/model.py gcnet/train_gcnet.py \
  experiments/bilstm_ablation/run_factorial.py \
  experiments/bilstm_ablation/summarize.py
git diff --check
```

预期：现有 conditional GPU skip 之外全部 PASS，无编译或 whitespace 错误。

- [ ] **步骤 2：同步干净 commit 到 biggpu**

远程训练仓库必须位于新目录/分支，`git status --porcelain` 为空；核验官方
`gcnet-official` Python 3.8、Torch 1.8、PyG 2.0.1 和四个数据根。

- [ ] **步骤 3：运行四数据集 one-batch forward/backward**

四格 × 四数据集各取一个真实 batch，检查 loss/gradient 有限、shape 正确、bypass 模块无梯度、split/sample/label fingerprint 被记录。

- [ ] **步骤 4：运行 16 个 short smoke**

四格 × 四数据集 × rate0.7 × seed66，1–2 epochs、`--allow-short-run`；只检查运行和 archive，不进入性能汇总。

- [ ] **步骤 5：更新实验记录并提交**

记录命令、commit、环境、16/16 状态、数据 fingerprints 和任何限制；按 Lore 协议提交。

### 任务 6：可复用 pilot 与完整 screening

**文件：**
- 修改：`experiments/bilstm_ablation/EXPERIMENT.zh.md`
- 生成：外部 `/data2/yb/paper/experiments/gcnet_bilstm_factorial_20260825/`

- [ ] **步骤 1：运行 96-job pilot**

四格 × 四数据集 × rates `{0.0,0.7}` × seeds `{66,67,68}`，100 epochs；每张 GPU 最多 3 workers。运行期间不修改模型或超参数。

- [ ] **步骤 2：运行重复噪声诊断**

重跑 IEMOCAPSix/original/rate0.0/seed66 和 CMUMOSI/original/rate0.0/seed66 到独立 repeat namespace，计算 W-F1 绝对漂移；目标 `<0.005`。

- [ ] **步骤 3：机械检查扩展门**

要求 96/96 成功、归档完整、四格 mask/split/label hash 配对、无坍塌、on/on 落入预注册复现容差，且重复漂移 `<0.005`。门只决定工程可靠性，不依据某 arm 是否提升。

- [ ] **步骤 4：续跑剩余 formal jobs**

在同一 output root 运行四数据集 × 四格 × 八率 ×五 seeds；runner 自动跳过已完成 pilot cell，最终达到 640/640。

- [ ] **步骤 5：汇总和同步结果**

运行 `summarize.py` 生成 JSON/中文 Markdown，rsync 全部 manifest/status/log/NPZ/summary 回本地 `/data2/yb/paper/experiments/gcnet_bilstm_factorial_20260825/`。

- [ ] **步骤 6：写入最终消融判定并提交**

按每数据集分别报告八档曲线、三个主效应和交互；明确 fold-5 screening、CUDA noise、五 seed 统计限制及下一步是否值得与 CP-LECC 交叉。按 Lore 协议提交文档。
