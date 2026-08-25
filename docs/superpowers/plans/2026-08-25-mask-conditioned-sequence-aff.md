# Mask-Conditioned Sequence AFF 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development 逐任务实现；每个任务按 TDD、规格审查、代码质量审查完成。

**目标：** 将 Dai et al. AFF 迁移为 mask-conditioned Sequence AFF，唯一替换 GCNet temporal/speaker 分支的直接相加，并完成 IEMOCAP-6 八档 paired A/B。

**架构：** 新模块用 utterance-local Linear context、conversation masked-mean global context 和 6D missing pattern 生成互补通道权重；零初始化和完整模态旁路保证初始/ATV direct-addition 等价。默认模型仍使用 addition。

**技术栈：** Python 3.8、PyTorch 1.8、NumPy、PyG 2.0.1、unittest、V100。

---

### 任务 1：Sequence AFF 核心模块

**文件：**
- 创建：`gcnet/sequence_aff.py`
- 创建：`tests/test_sequence_aff.py`

- [ ] 先写七模式编码、shape、ATV forward/backward 等价、全模式零初始化等价、padding 不影响 global mean、非零参数产生内容/pattern/global 依赖的失败测试。
- [ ] 运行 `PYTHONPATH=gcnet python -m unittest tests.test_sequence_aff -v`，确认模块不存在而 RED。
- [ ] 实现 `MaskConditionedSequenceAFF(channels,reduction=4,pattern_dim=6)`：Linear+LayerNorm+ReLU local/global，两条输出 Linear 零初始化；用 `umask.T` 做 masked mean，用安全 ATV mask 替换 padded mask 后复用 `encode_missing_patterns`。
- [ ] 实现 `base + incomplete*(2*(w*x+(1-w)*y)-base)`；严格验证输入、mask、umask shape 和二值合法性。
- [ ] 运行测试、`py_compile`、diff-check并按 Lore 提交。

### 任务 2：GraphModel 与训练入口

**文件：**
- 修改：`gcnet/model.py`
- 修改：`gcnet/train_gcnet.py`
- 修改：`tests/test_model_mpfilm_integration.py`
- 修改：`tests/test_training_protocol.py`

- [ ] 先写 `branch_fusion=addition|mask_sequence_aff` 默认值、模块构造、默认参数/RNG/output 等价、完整模态整模型等价、CLI/文件名/NPZ provenance 的失败测试。
- [ ] 从 forked CPU RNG 初始化 Sequence AFF，默认 `addition`；在 `hidden1/hidden2` 之后只选择 direct addition 或新模块。
- [ ] CLI 增加 Python3.8-compatible `--branch-fusion` choices；文件名追加 `_branchfusion:<mode>`；NPZ 保存 fusion choice、stored total 和 selected path count。
- [ ] 运行 Sequence AFF、BiLSTM、图集成、训练协议全套聚焦测试并按 Lore 提交。

### 任务 3：复用 IEMOCAP-6 锁定 runner 与汇总

**文件：**
- 修改：`experiments/mpfilm_iemocap6/run_locked_ab.py`
- 创建：`experiments/sequence_aff_iemocap6/__init__.py`
- 创建：`experiments/sequence_aff_iemocap6/summarize.py`
- 创建：`tests/test_sequence_aff_runner.py`
- 创建：`tests/test_sequence_aff_summary.py`

- [ ] runner 新增 `sequence_aff` arm：graph variant 固定 original、branch fusion 为 mask_sequence_aff；既有 arms 显式 addition；不改变其他命令。
- [ ] 测试 2 arms × 8 rates × 5 seeds = 80 unique jobs、配对 mask、命令 identity 和不可变 resume。
- [ ] 汇总器验证 trusted NPZ 的 dataset/fold/rate/seed/context/graph/fusion/mask hash/100 epochs，计算每 rate 和八档宏平均的 paired effect、SD、win count、t/Wilcoxon、collapse。
- [ ] 跑聚焦与现有 runner tests，按 Lore 提交。

### 任务 4：远程 paired A/B

**生成目录：**
- `/data2/yb/paper/experiments/sequence_aff_iemocap6_20260825/`

- [ ] 将干净 commit 克隆到 biggpu 新目录，核验 official Python3.8/Torch1.8/PyG2.0.1 和数据 hashes。
- [ ] 先运行 Original/Sequence-AFF 的 rate0.0、seed66 short operational smoke；只判断归档和等价方向。
- [ ] 运行 Original + Sequence-AFF，rates0.0--0.7、seeds66--70、fold5、100 epochs，共80 jobs；4 GPUs×3 workers。
- [ ] 验证80/80、100 epochs、单NPZ、无lock、配对mask hash；运行汇总器。
- [ ] rsync结果回本地并写 `experiments/sequence_aff_iemocap6/EXPERIMENT.zh.md`，明确 fold5 screening 和 CUDA noise 限制。
