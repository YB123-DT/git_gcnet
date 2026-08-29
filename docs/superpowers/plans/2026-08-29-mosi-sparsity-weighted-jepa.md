# MOSI Budget-Preserving Sparsity-Weighted JEPA 实现计划

> 单一目标：在不改模型、mask 和推理的前提下，将固定 JEPA 预算从低缺失 rate
> 平移到高缺失 rate，并用严格配对 validation 判断是否稳定有效。

## 任务 1：TDD 锁定权重与默认兼容

- [x] 测试 `uniform` 对所有 rates 返回 `1.0`。
- [x] 测试 `sparsity-budget` 单调递增，且 `0.1--0.7` 平均精确为 `1.0`。
- [x] 测试 rate 权重只乘 JEPA 项，不乘 task loss 或八-rate `1/8` 聚合。
- [x] 测试非法 mode 失败。

## 任务 2：最小实现

- [x] 在 `TrainConfig`、CLI、config、metrics 中加入 `jepa_rate_weighting`，默认
  `uniform`。
- [x] 在 `train_epoch()` 的现有 `task + lambda_J * JEPA` 位置应用权重。
- [x] 不改变 `missing_m3_loss()`、模型 forward 或 evaluation。

## 任务 3：复用现有 runner

- [x] 将 `sparsity-budget` 注册为唯一 treatment，目录固定为
  `sparsity-weighted-jepa/seed_*`。
- [x] command、manifest、resume、candidate/control audit 记录该字段。
- [x] 禁止与其他已注册 treatment 组合；要求 direct deterministic controls。

## 任务 4：验证与实验

- [x] 运行相关完整测试、`git diff --check`、`py_compile`。
- [x] 运行 seeds 66--68 validation-only screen，不重跑 controls。
- [x] 同步 config/history/metrics/summary；审计 100 epochs、无 test、唯一变量和 gate。
- [x] 通过才扩 seeds 69/70；失败则关闭该路线并记录。

结果：overall `-0.2526` point，high-missing `-0.5838` point，miss-0
`+0.0031` point，`1/3` seeds 正向；正式关闭，不读取 test。
