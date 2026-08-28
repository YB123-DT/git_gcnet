# MOSI All-Rates-Per-Batch 实现计划

**目标：** 在不增加模型参数和 optimizer step 数的条件下，让每个 MOSI conversation batch 覆盖全部八个 missing rates，判断 rate under-exposure 是否导致当前 85.x 上限。

## 任务 1：TDD 锁定训练生命周期

- [ ] 在 `tests/test_missing_m3.py` 先写 all-mode 红灯测试；
- [ ] 验证八 rate 顺序、loss averaging、一次 optimizer step、一次 EMA update、一次 teacher encoding；
- [ ] 验证默认 cyclic 行为不变；
- [ ] 运行 targeted tests 观察预期失败。

## 任务 2：最小实现

- [ ] 为 `TrainConfig` 和 CLI 增加 `train_rate_mode`；
- [ ] 将单 batch 更新抽成可测试 helper；
- [ ] 实现 cyclic/all 两条明确路径，不修改 evaluation；
- [ ] 记录 `optimizer_steps` 与 rate counts；
- [ ] targeted 与完整测试转绿。

## 任务 3：远程验证与 seed 66

- [ ] 同步最小代码到 biggpu；
- [ ] 远程完整 tests；
- [ ] 1-epoch GPU smoke，验证每个 rate 两个 batch且仅两个 optimizer/EMA steps；
- [ ] 运行 seed66 100 epochs；
- [ ] 按 miss0 87.0 和 nonzero delta -0.5 门槛决定是否扩种子。

## 任务 4：审计与交付

- [ ] 重算 NPZ 指标与 paired mask hashes；
- [ ] 两阶段代码/结果审查；
- [ ] 更新实验报告；
- [ ] Lore commit 并推送 GitHub。

