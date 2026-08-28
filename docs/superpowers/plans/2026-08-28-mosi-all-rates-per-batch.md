# MOSI All-Rates-Per-Batch 实现计划

**目标：** 在不增加模型参数和 optimizer step 数的条件下，让每个 MOSI conversation batch 覆盖全部八个 missing rates，判断 rate under-exposure 是否导致当前 85.x 上限。

**执行状态：** 已完成。seed 66 的 miss0 W-F1 从 85.69 提升到 86.56，nonzero-rate mean 提升 1.23，但 miss0 未达到 87.0 扩展门槛，因此不扩 seeds 67–70。

## 任务 1：TDD 锁定训练生命周期

- [x] 在 `tests/test_missing_m3.py` 先写 all-mode 红灯测试；
- [x] 验证八 rate 顺序、loss averaging、一次 optimizer step、一次 EMA update、一次 teacher encoding；
- [x] 验证默认 cyclic 行为不变；
- [x] 运行 targeted tests 观察预期失败。

## 任务 2：最小实现

- [x] 为 `TrainConfig` 和 CLI 增加 `train_rate_mode`；
- [x] 以最小分支实现可测试的 batch 生命周期，未额外抽象 helper；
- [x] 实现 cyclic/all 两条明确路径，不修改 evaluation；
- [x] 记录 `optimizer_steps` 与 rate counts；
- [x] targeted 与完整测试转绿。

## 任务 3：远程验证与 seed 66

- [x] 同步最小代码到 biggpu；
- [x] 远程完整 tests；
- [x] 1-epoch GPU smoke，验证每个 rate 两个 batch且仅两个 optimizer/EMA steps；
- [x] 运行 seed66 100 epochs；
- [x] 按 miss0 87.0 和 nonzero delta -0.5 门槛决定是否扩种子。

## 任务 4：审计与交付

- [x] 重算 NPZ 指标与 paired mask hashes；
- [x] 两阶段代码/结果审查；
- [x] 更新实验报告；
- [x] Lore commit 并推送 GitHub。
