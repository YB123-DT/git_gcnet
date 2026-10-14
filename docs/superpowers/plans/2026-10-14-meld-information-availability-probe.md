# MELD Information Availability Probe 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。

**目标：** 在冻结的 MELD cfg84 no-JEPA causal Flat seed66 checkpoint 上，比较 Local、OSRAM Memory、严格历史摘要及其同维 donor 对照，区分历史信息弱、写入损失和读出未利用三种解释。

**架构：** 复用现有 MELD legacy mask、模型构建和 Flat capture，只新增离线特征整理、严格因果历史摘要、split-local donor 匹配、统一七分类 Probe 与结果聚合。每个 missing rate 的 backbone 只扫描 train/test 各一次；Probe 不向 backbone 反传。

**技术栈：** Python、PyTorch、NumPy、scikit-learn metrics、pytest、现有 GCNet/OSRAM loader 与 checkpoint。

---

## 任务 1：锁定严格因果特征和 donor 语义

**文件：**
- 新建：`tests/test_meld_information_features.py`
- 新建：`experiments/osram_meld_information_availability_20261014/__init__.py`
- 新建：`experiments/osram_meld_information_availability_20261014/features.py`

- [ ] 先写失败测试：两段对话、重复 speaker、首句和不同 availability，断言 all-history/same-speaker mean 只使用 `s<t`，首句为零，presence/count 正确。
- [ ] 写失败测试：inactive Gap 和 padding 置零，forward 512 保留，backward half 不进入缓存。
- [ ] 写失败测试：donor 必须来自同 split 的不同 conversation，并匹配 availability、all-history presence、same-speaker-history presence；匹配过程不能读取 label。
- [ ] 写失败测试：无 donor 的行在 real/donor 两侧同时排除，返回精确 coverage 和稳定映射。
- [ ] 运行 `pytest -q tests/test_meld_information_features.py`，确认失败原因是模块尚未实现。
- [ ] 最小实现 `strict_past_summaries`、`mask_forward_memory`、`match_donors` 和 cache schema 验证。
- [ ] 重新运行测试直至通过。
- [ ] 提交：`Preserve causal history semantics in MELD availability probes`。

## 任务 2：实现统一分类 Probe 与同维对照

**文件：**
- 新建：`tests/test_meld_information_probe.py`
- 新建：`experiments/osram_meld_information_availability_20261014/probe.py`

- [ ] 先写失败测试：构造七个 real 输入及六个 exact-width donor 输入；`Local` 不产生 donor 版本，`Local+*` 中 current Local 不得被替换。
- [ ] 写失败测试：normalizer 只从 train 拟合；测试集中极端值不能改变 train 统计；常数维数值稳定。
- [ ] 写失败测试：统一网络为 `LayerNorm -> Linear(64) -> GELU -> Linear(7)`，所有变体共享训练超参数且产生有限梯度。
- [ ] 写失败测试：ACC/W-F1 从保存预测可重算，paired real/donor 使用同一 row IDs 和覆盖集合。
- [ ] 运行 `pytest -q tests/test_meld_information_probe.py`，确认红灯。
- [ ] 最小实现输入构造、训练、best test-oracle checkpoint 保存、预测保存、指标及 paired delta。
- [ ] 重新运行测试直至通过。
- [ ] 提交：`Measure MELD history information against exact-width donor controls`。

## 任务 3：接入冻结 MELD checkpoint 提取与运行状态

**文件：**
- 新建：`tests/test_meld_information_runner.py`
- 新建：`experiments/osram_meld_information_availability_20261014/run.py`
- 新建：`experiments/osram_meld_information_availability_20261014/README.md`
- 新建：`experiments/osram_meld_information_availability_20261014/RESULT.md`

- [ ] 先写失败测试：legacy mask materialization 与源评测一致，valid utterance ID 唯一，train/test conversation 无交叉。
- [ ] 写失败测试：抽取结果包含 Local 256、Base/Gap 各 forward 512、metadata；首句 Memory/RawHistory 为零；inactive Gap 严格为零。
- [ ] 写失败测试：模型参数全冻结，抽取前后 state hash 一致；保存预测与源 checkpoint 指标一致。
- [ ] 写失败测试：输出目录不可覆盖，`STATUS.json` 在成功/失败时原子更新，记录源 config/checkpoint/data/code hash。
- [ ] 运行 `pytest -q tests/test_meld_information_runner.py`，确认红灯。
- [ ] 复用 `FlatReadoutCapture`、`materialize_legacy_masks`、`_build_model` 和现有 loader，完成每 rate 一次 train/test 扫描。
- [ ] 加入 `--check-only`：仅一个 rate、少量 batch、两 epoch Probe，用于真实 checkpoint smoke。
- [ ] README 明确 `INTERNAL DIAGNOSTIC ONLY`、Test-oracle 来源、无 backbone 训练及运行命令。
- [ ] 重新运行三组 focused tests 与相关 MELD legacy regression tests。
- [ ] 提交：`Make frozen MELD information probes reproducible from retained checkpoints`。

## 任务 4：实现严格聚合和 A/B/C 判读

**文件：**
- 新建：`tests/test_meld_information_analysis.py`
- 新建：`experiments/osram_meld_information_availability_20261014/analyze.py`
- 修改：`experiments/osram_meld_information_availability_20261014/RESULT.md`

- [ ] 先写失败测试：每 rate、每 Probe seed 的 ACC/W-F1；8-rate mean；0.5/0.6/0.7 high mean；三 seed mean/sample std。
- [ ] 写失败测试：real-minus-donor 只在 paired coverage 上计算，不能拿 full-cohort 指标相减。
- [ ] 写失败测试：按设计文档规则产生 Case A/B/C/mixed 描述；Memory-only 高但 Local+Memory 无条件增益不能标成 C。
- [ ] 运行 `pytest -q tests/test_meld_information_analysis.py`，确认红灯。
- [ ] 最小实现 `SUMMARY.json` 和 `RESULT.md` 生成器，引用 P0 的 MELD Base/Gap 近零增益但不重跑 P0/P1。
- [ ] 重新运行全部新测试。
- [ ] 提交：`Turn MELD probe outputs into falsifiable availability diagnoses`。

## 任务 5：真实 checkpoint smoke 与单次授权运行

**文件：**
- 修改：`experiments/osram_meld_information_availability_20261014/RESULT.md`

- [ ] 运行 `python -m py_compile` 和所有新增 pytest；确认 git diff 仅含本实验与计划。
- [ ] 提交并推送实现，随后用 immutable git archive 同步到 biggpu；核对 GPU UUID，绝不使用宿主机 GPU4。
- [ ] 运行 `--check-only --rates 0.0`，验证源预测 parity、冻结 hash、严格历史、inactive Gap、donor coverage、有限 Probe gradient 和保存/恢复。
- [ ] Smoke 通过后启动且仅启动 MELD seed66 的 0.0--0.7 正式内部诊断；三种 Probe 初始化，backbone 每 split/rate 只扫描一次。
- [ ] 记录 PID、GPU UUID、命令、日志、代码 commit、checkpoint/data hash、峰值显存和耗时。
- [ ] 运行 analyzer，检查 13 个变体、8 rates、3 Probe seeds 均完整；任何缺项不得用部分均值冒充完成结果。
- [ ] 更新 `RESULT.md`，明确 P0/P1 复用、A/B/C 结论边界和停止规则。
- [ ] 最终运行测试、重算 summary hash、提交并推送结果。

## 完成标准

- [ ] 默认训练/推理代码未修改，原 OSRAM 行为不受影响。
- [ ] MELD seed66 八率冻结缓存与 13 个 Probe 变体完整。
- [ ] donor 匹配无标签、同 split、跨 conversation、严格 paired。
- [ ] 所有指标可由保存预测重算。
- [ ] 结果只作为 `INTERNAL DIAGNOSTIC ONLY`，不扩展数据集、backbone seed 或新模块。
