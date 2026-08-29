# CMU-MOSI Hidden–Window 扫描设计

## 目标

在当前正式 Slot Missing-M3 配置中固定 `time-attention=False`，判断 GCNet
隐藏维度与图窗口是否解释 MOSI 的性能缺口。

## 唯一变量

- `hidden ∈ {50, 100, 200}`；
- 对称窗口 `windowp=windowf ∈ {1, 2, 3, 4}`；
- 共 12 个配置。

其余配置锁定为 CMUMOSI、fold 1、Slot、Regression、双图分支、all-rates-per-batch、
100 epochs、batch size 32、learning rate `5e-4`、weight decay `1e-5`、JEPA
weight `0.1`。每个训练任务产生一个 checkpoint，并测试全部八个 missing rates。

## 并发与阶段

第一阶段运行 seeds 66、67、68。每张 GPU 负责一个 seed 的 12 个任务，但每卡最多
同时运行 3 个任务；因此分四个 wave 执行，每个任务限制为两个 CPU 线程。最初的
12任务/卡启动试验在动态峰值 batch 上发生 OOM，不能作为正式调度方式。

每个任务输出到独立目录。存在完整 `metrics.json` 时恢复运行会跳过该任务；失败任务保留
日志和状态并允许单独重跑。

## 选择规则

只使用三个 seed 的 `best_validation_mean_weighted_f1` 均值对 12 个配置排序。
测试集八 rate、miss0 和 high-missing 只在配置顺序锁定后汇总，不参与参数选择。

若最佳配置相对 `hidden=200, window=2` 的 validation 均值没有明确提高，则保留 Control；
若明确提高，再补 seeds 69、70，不重新扫描 `time-attention=True`。

## 完整性检查

- 恰好 36 个第一阶段任务；
- 每张 GPU 恰好分配 12 个任务，每个 wave 不超过 3 个；
- 每个 seed 恰好覆盖 12 个不重复的 `(hidden, window)`；
- 命令不得包含 `--time-attn`；
- 每个任务包含 `--num-threads 2`；
- 输出目录和日志路径不碰撞。
