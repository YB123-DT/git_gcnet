# MPFiLM-RGCN 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在官方 GCNet 中实现 complete-preserving missing-pattern FiLM 第一层，并完成 IEMOCAP-6 fold 5 的固定-mask A/B 实验。

**架构：** 新卷积保留 PyG RGCN 的 relation weight、relation-wise mean、root 和 bias，只在不完整边上加入 source-pattern correction 与 target-conditioned FiLM。原始三位 mask 通过 `GraphModel` 传到 temporal/speaker 两个分支，并按 `batch_graphify` 的 conversation-major 顺序展平。

**技术栈：** Python 3.10、PyTorch 2.2.2、PyG 2.4.0、`unittest`、NumPy、IEMOCAP-6。

---

## 文件职责

- `gcnet/mpfilm_rgcn.py`：pattern 编码、节点顺序展平和 `MissingPatternFiLMRGCNConv`。
- `gcnet/model.py`：选择卷积变体，将 mask 送入两个图分支。
- `gcnet/train_gcnet.py`：CLI 变体/fold/mask-bank 参数、固定 mask 加载、结果元数据。
- `gcnet/mask_bank.py`：按 conversation/utterance 构建并读取固定 mask bank。
- `tests/test_mpfilm_rgcn.py`：九类模块与对齐测试。
- `tests/test_mask_bank.py`：固定性、跨 arm 共享与 0.7 pattern 测试。
- `experiments/mpfilm_iemocap6/run_locked_ab.py`：三进程每卡调度和锁定命令生成。
- `experiments/mpfilm_iemocap6/EXPERIMENT*.md`：协议、状态与结果的中英记录。

### 任务 1：先锁定卷积行为

- [ ] 创建 `tests/test_mpfilm_rgcn.py`，先导入尚不存在的 `MissingPatternFiLMRGCNConv` 和 `encode_missing_patterns`，覆盖 forward/backward 等价、七模式、0.7、同质邻域、单邻居、参数量和设备。
- [ ] 运行 `PYTHONPATH=gcnet python -m unittest tests.test_mpfilm_rgcn -v`，预期因 `mpfilm_rgcn` 不存在而失败。
- [ ] 创建 `gcnet/mpfilm_rgcn.py`，实现以下公共接口：

```python
def encode_missing_patterns(node_mask: Tensor) -> tuple[Tensor, Tensor]: ...

class MissingPatternFiLMRGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations,
                 variant="full", parameter_budget=None): ...
    def forward(self, x, edge_index, edge_type, node_mask): ...
```

- [ ] 使用每个 relation 的 `scatter(..., reduce="mean")`，加入 root/bias；`pattern_weight` 和 `film_weight` 零初始化。
- [ ] 重跑测试，直至 CPU 全绿；有 CUDA 时执行 GPU FP32 forward/backward。
- [ ] 提交卷积与测试。

### 任务 2：锁定 GCNet mask 对齐和双分支集成

- [ ] 先写失败测试，构造不同长度的两个 conversation，验证：

```python
expected = torch.cat([mask[:lengths[b], b] for b in range(len(lengths))])
actual = flatten_valid_node_masks(mask, lengths)
torch.testing.assert_close(actual, expected)
```

- [ ] 在 `gcnet/mpfilm_rgcn.py` 实现 `flatten_valid_node_masks`，拒绝 padded reshape 和 000。
- [ ] 修改 `GraphNetwork`：`original` 保留 PyG 层，其他变体构造 MPFiLM；forward 接收 `node_mask`。
- [ ] 修改 `GraphModel.forward(inputfeats, input_features_mask, qmask, umask, seq_lengths)`，只展平一次并将同一 node mask 传给 temporal/speaker 分支。
- [ ] 修改训练调用；`reccls_flag` 两次前向均传原始 availability mask。
- [ ] 运行全部 unittest 并提交。

### 任务 3：固定 mask bank

- [ ] 先创建 `tests/test_mask_bank.py`，验证相同 seed/rate/utterance 得到相同 mask，不同 model arm 读取完全相同数组，0.7 只含 A/T/V，所有有效 utterance 至少一个模态。
- [ ] 创建 `gcnet/mask_bank.py`，复用官方 `random_mask` 算法生成 `{vid: [length,3]}`，保存 `.npz` 和包含 seed/rate/生成器版本的 JSON manifest。
- [ ] 在训练脚本加入 `--mask-bank-root`、`--mask-seed`，按当前 batch 的 `vidnames` 重建 `[T,B,3]`，并用它同时 mask host/guest 的被选 utterance。
- [ ] 增加 `--fold-index`，取值 1..5；正式实验固定为 5，避免无意运行全部 fold。
- [ ] 运行 mask-bank 与集成测试并提交。

### 任务 4：实验 CLI 与 smoke

- [ ] 加入 `--graph-conv-variant {original,pattern_only,full,content_film_control}`、`--output-dir`、`--allow-short-run`；短程模式绕过官方 `epochs>=60` 的保存断言但必须在输出标注 `SMOKE_ONLY=true`。
- [ ] 结果保存 variant、git commit、fold、train seed、mask seed、missing rate、parameter counts 和 mask-bank hash。
- [ ] 建立本地数据软链接并以 IEMOCAP-6 fold 5、missing rate 0.7、1 epoch 运行 Original/Full smoke。
- [ ] 检查 loss 有限、六类标签/prediction 统计可读取、新增参数得到非零梯度、Complete 单测仍通过。
- [ ] smoke 失败时只修复导致失败的单一变量并重跑；提交可执行 harness。

### 任务 5：正式 A/B 与分析

- [ ] 先按 `Original vs Full`、fold 5、8 missing rates、5 seeds 生成 80 个锁定任务；每 GPU 同时最多三个进程。
- [ ] 优先运行 0.0 和 0.7 的 2-seed 门控批次。若 Complete 结果不匹配预期方差或出现 NaN/单类预测，停止扩展并诊断。
- [ ] 门控通过后启动全部 80 个任务；训练期间只监控，不修改模型、损失、mask 或选择规则。
- [ ] 汇总 weighted F1/accuracy 的逐 seed 值、均值、标准差、Full-Original 配对差、类别覆盖和坍塌判据。
- [ ] 只有 Full 在主 A/B 显示稳定信号后，才运行 `pattern_only` 与 `content_film_control`。

## 计划自检

- 规格的五个 conditioning 输入、complete invariant、两个第一层、固定 mask 和九类测试均有对应任务。
- 第一轮只比较 Original/Full，避免在主结论前消耗消融预算。
- smoke 与正式结果明确隔离，任何 smoke F1 都不能写成研究结论。
