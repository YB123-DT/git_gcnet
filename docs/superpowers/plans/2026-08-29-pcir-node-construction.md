# PCIR 图前联合节点构造实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 与 test-driven-development 逐任务实现。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 Original Slot node 上加入 pattern-conditioned unary calibration 与 observed-pair interaction residual。

**架构：** 新模块只接收 observed Student latents、availability 和 padding mask，返回零初始化的 `[L,B,256]` residual。MissingM3GraphModel 在图前执行 `slot_node + residual`，其余路径保持不变。

**技术栈：** PyTorch、pytest、现有 Missing-M3 trainer、biggpu V100。

---

### 任务 1：锁定 PCIR 单元行为

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`

- [ ] 编写失败测试，覆盖七 pattern、padding、missing leakage、单/双/三模态 pair 激活和初始化零输出。
- [ ] 远程运行 focused tests，确认只因 `PatternConditionedInteractionResidual` 不存在而失败。
- [ ] 实现 `PatternConditionedInteractionResidual(latent_dim=256, pair_embedding_dim=32, pair_rank=64, residual_hidden_dim=128)`。
- [ ] 运行 focused tests 至绿灯。

### 任务 2：接入模型并锁定兼容性

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/model.py`

- [ ] 编写失败测试：default state/output/RNG 不变、shared initialization 相同、初始 treatment 输出等于 Control、完整 backward。
- [ ] 在 `MissingM3GraphModel` 末尾新增 `node_interaction_residual=False`；只在开启时实例化 PCIR。
- [ ] 在 Slot node 与 `encode_hidden` 之间加入 residual；对冲突 variant 显式报错。
- [ ] 运行 focused tests 至绿灯。

### 任务 3：CLI/config 与完整验证

**文件：**
- 修改：`tests/test_missing_m3.py`
- 修改：`gcnet_missing_m3/train_gcnet.py`

- [ ] 先写 `--node-interaction-residual` CLI/config 红灯测试。
- [ ] 在 `TrainConfig` 尾部追加字段并透传模型；默认关闭。
- [ ] 运行 Missing-M3、text-LoRA、PLCI 完整远程测试；预期全部通过。
- [ ] 运行一次真实 GPU forward/backward smoke，记录参数量；不运行重复 1-epoch 保存测试。

### 任务 4：MOSI 五种子正式 A/B

**文件：**
- 创建：`experiments/missing_m3_mosi_pcir_node_20260829/EXPERIMENT.md`
- 创建：`experiments/missing_m3_mosi_pcir_node_20260829/results/SUMMARY.json`

- [ ] GPU 0/1/2/3/5 并行运行 seeds66--70、`lr=5e-4`、8 rates、100 epochs。
- [ ] 继承相同 seeds/rates 的 Slot `5e-4` Control，不重跑。
- [ ] 独立重算 40 个 NPZ W-F1，核验 40 个 mask SHA。
- [ ] 按预注册门槛判断 PASS/FAIL，不追加救援模块。
- [ ] 按 Lore protocol 提交并推送 `github feature/m3-jepa-gcnet`。
