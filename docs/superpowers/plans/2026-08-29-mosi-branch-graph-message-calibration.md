# MOSI Branch-Specific Graph-Message Calibration 实现计划

## 任务 1：TDD 锁定默认与校准合同

- [x] 默认/显式 `none` 的 keys、参数、RNG、forward 精确一致。
- [x] treatment 零初始化时 graph message 与 raw message 一致。
- [x] 手动设置 alpha 后符合逐通道 LayerNorm residual 公式。
- [x] 首次 backward 给 alpha 有限非零梯度。
- [x] 两分支 alpha 独立，正式配置只增加 100 个参数。

## 任务 2：最小模型贯通

- [x] `GraphNetwork` 在 conv2 后应用可选校准。
- [x] `GraphModel` 与 `MissingM3GraphModel` 将开关传给两分支。
- [x] `TrainConfig`、CLI、config、metrics 显式记录，默认 `none`。

## 任务 3：复用 runner

- [x] 注册唯一目录 `branch-graph-message-calibration/seed_*`。
- [x] command、manifest、resume、control audit 记录开关。
- [x] 禁止与其他 treatment 组合，强制 direct deterministic controls。

## 任务 4：验证与筛选

- [x] 运行相关完整测试、CUDA FP32 前后向、diff-check 与 py_compile。
- [x] 运行 seeds 66--68 validation-only，不重跑 controls。
- [x] 同步并审计；通过才补 69/70，失败则关闭本轮 graph/sequence 边界路线。

结果：overall `+0.0620` point，high-missing `-0.4704` point，miss-0
`+1.0748` points，`2/3` seeds 正；gate 失败且未读取 test。
