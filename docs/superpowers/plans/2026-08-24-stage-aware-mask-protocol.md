# Stage-aware paired mask protocol 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 恢复 GCNet 在固定缺失率下的训练 mask 多样性，同时保持跨方法、验证和测试严格可复现。

**架构：** 将现有单一 utterance mask bank 扩展为 stage-aware bundle：100 个训练 epoch bank、1 个 validation bank、1 个 test bank。训练循环显式传入 stage 和 epoch，不依赖随机调用顺序；保存 bundle 与子 bank 哈希。

**技术栈：** Python、NumPy、PyTorch、unittest、GCNet official environment。

---

### 任务 1：锁定 stage-aware bank 行为

**文件：**
- 修改：`tests/test_mask_bank.py`
- 修改：`gcnet/mask_bank.py`

- [ ] **步骤 1：编写失败测试**

新增测试，要求相同 seed 的 bundle 字节级一致、相邻训练 epoch 至少一个有效 utterance pattern 不同、validation/test 不同、每个 rate=0.7 bank 的 realized rate 为 2/3。

- [ ] **步骤 2：验证红灯**

运行：`PYTHONPATH=gcnet python -m unittest tests.test_mask_bank`

预期：缺少 stage-aware bundle API，测试失败。

- [ ] **步骤 3：实现最小 bundle API**

在 `mask_bank.py` 复用 `build_mask_bank`，以稳定派生 seed 创建 `train[epoch]`、`validation`、`test`，并计算 bundle/constituent SHA256；不得改变 legacy mask 生成公式。

- [ ] **步骤 4：验证绿灯并提交**

运行上述测试，预期全部通过；提交 mask bundle 与测试。

### 任务 2：训练循环按 stage/epoch 消费 bank

**文件：**
- 修改：`tests/test_training_protocol.py`
- 修改：`gcnet/train_gcnet.py`

- [ ] **步骤 1：编写失败测试**

使用记录型 bank selector 验证 epoch 1/2 的训练调用分别读取对应 bank，validation/test 读取固定 stage bank，且三次调用不依赖全局 RNG 调用顺序。

- [ ] **步骤 2：验证红灯**

运行：`PYTHONPATH=gcnet python -m unittest tests.test_training_protocol`

预期：当前训练函数只接受一个平坦 bank，测试失败。

- [ ] **步骤 3：实现显式选择**

为 `train_or_eval_model` 增加已选择的 stage bank 输入；epoch 循环在调用前选择 `train[epoch]`、`validation`、`test`。归档写入 bundle manifest，命令行和其他训练配置不变。

- [ ] **步骤 4：验证绿灯和回归并提交**

运行 mask、training、CP-LECC、模型集成测试及完整 unittest discovery；预期全部通过。

### 任务 3：biggpu Original 恢复门

**文件：**
- 创建：`experiments/cp_lecc_iemocap6/PROTOCOL_RECOVERY.zh.md`

- [ ] **步骤 1：同步并验证官方环境**

同步干净 Git 快照到 biggpu；运行官方 Python 的模型导入、mask 测试和单步 CUDA 前向。

- [ ] **步骤 2：运行 10 个恢复任务**

运行 Original、rates 0.5/0.7、seeds 66--70、fold 5、100 epochs；GPU 0--3、每卡最多 3 个并发。

- [ ] **步骤 3：审核恢复门**

验证 10/10 success、100 epochs、唯一 archive、coverage=6、stage/bundle hashes；与旧同 seed 均值 0.608924/0.611208 比较，每档下降不得超过 0.015。

- [ ] **步骤 4：同步与记录**

把所有日志、状态、归档和恢复结论同步回 `/data2/yb/paper` 并写入恢复文档。

### 任务 4：通过恢复门后重跑方法比较

**文件：**
- 修改：`experiments/cp_lecc_iemocap6/PROTOCOL_RECOVERY.zh.md`

- [ ] **步骤 1：条件检查**

仅当任务 3 两档均通过恢复门时继续；否则停止方法训练并诊断协议。

- [ ] **步骤 2：运行 Full 与 CP-LECC**

使用与 Original 完全相同的 stage-aware bundles 运行 20 个任务。

- [ ] **步骤 3：汇总**

报告逐 rate 均值、标准差、逐 seed paired delta、wins、coverage 和 mask hash 一致性；不得复用单一固定训练 mask 的旧结论。
