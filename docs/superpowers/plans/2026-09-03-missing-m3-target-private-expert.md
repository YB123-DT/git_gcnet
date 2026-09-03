# Missing-M3 Target-Private Expert 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:test-driven-development 在当前隔离工作树内逐项实现。

**目标：** 在现有 Missing-M3 六方向预测器中增加目标模态私有的低秩残差专家，缓解 A/T/V 预测任务共享专家造成的负迁移，同时保持默认配置完全不变。

**架构：** 共享 MMoE 继续提取跨模态公共结构；每个目标模态拥有一个 `Linear(d,r) -> GELU -> Linear(r,d)` 私有专家，其输出在 target-specific head 之前加入回归和对比分支。末层零初始化；`rank=0` 时不实例化新参数，保持旧 checkpoint、state-dict 和推理路径。

**技术栈：** Python、PyTorch、pytest、现有 Missing-M3/GCNet 训练器。

---

### 任务 1：锁定私有专家行为

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`

- [x] 添加测试：默认 `rank=0` 不出现私有专家 key，输出与原模型一致。
- [x] 添加测试：`rank=32` 新增三个低秩私有专家，参数增量为 `3 * 2 * d * r`。
- [x] 添加测试：零初始化时输出与共享 MMoE 完全一致。
- [x] 添加测试：目标 A 的损失只更新 A 私有专家，不更新 T/V 私有专家。
- [x] 实现 `TargetPrivateExpertResidual` 并接入 `DualGateTopKMMoE`。

### 任务 2：贯通模型与训练配置

**文件：**
- 修改：`gcnet_missing_m3/model.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`
- 修改：`tests/test_missing_m3.py`

- [x] 将 `target_private_rank` 从 `TrainConfig`、CLI、`MissingM3GraphModel` 传到 MMoE。
- [x] 添加 `--target-private-rank`，默认值为 `0`，拒绝负数。
- [x] 验证 rank 0 的配置序列化和旧行为不变。

### 任务 3：记录独立实验版本并交付

**文件：**
- 创建：`experiments/missing_m3_target_ple_20260903/README.md`

- [x] 记录模块名称、作用、精确插入位置、默认/处理配置和后续正式实验协议。
- [x] 运行定向单元测试，不运行 smoke 或训练。
- [x] 检查 diff、提交 Lore commit，并推送 `feature/missing-m3-target-ple` 到 GitHub。
