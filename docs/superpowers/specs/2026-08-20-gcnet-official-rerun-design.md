# GCNet 官方协议四数据集重跑设计

## 目标

在同一代码、同一 Python 环境、同一随机性控制和同一缺失掩码调度下，重新运行 GCNet baseline 与 GCNet+JEPA，并采用原 GCNet 的评估拓扑输出可配对审计的结果。

## 固定实验矩阵

- 数据集：`IEMOCAPFour`、`IEMOCAPSix`、`CMUMOSI`、`CMUMOSEI`。
- 方法：`baseline`、`jepa`。
- 缺失率：`0.0` 至 `0.7`，步长 `0.1`。
- 种子：`66` 至 `75`。
- IEMOCAP：只运行 `fold=5`，即 Session 5 测试协议。
- 总配置数：`4 × 2 × 8 × 10 = 640`。
- 解释器：`/data2/yb/reproduction_envs/gcnet-official/bin/python`。
- GPU：只使用 `0,1,2,3,5`，排除 GPU 4；每卡最多三个并发任务。

## 官方评估拓扑

训练器增加 `official` 与 `strict` 两种显式协议，正式重跑默认 `official`：

- IEMOCAP official：四个 session 训练，held-out Session 5 同时作为 validation 和 test；每个 epoch 都执行 validation 和 test；按 validation Weighted-F1 最优 epoch 报告该 epoch 已计算的 test 结果。
- MOSI/MOSEI official：保留官方 train/validation/test 划分；每个 epoch 都执行 validation 和 test；同样按 validation 最优 epoch选择 test 结果。
- strict：保留当前训练/内部验证/测试一次的实现，只用于泄漏诊断，不进入本轮正式结果。

test 指标不能参与模型选择。official 模式虽然每轮测试，但最终 epoch 仅由 validation 决定。

## 公平配对与随机性

Baseline 和 JEPA 必须共享：初始化 checkpoint、数据顺序、缺失掩码、训练种子、公共稳定项和全部非方法参数。official 模式的 validation/test 掩码按 epoch 确定性变化，以保留官方每轮重新评估的拓扑，同时让 paired run 可复现。JEPA 只允许增加其方法专属损失与模块。

## 证据与恢复

每个 run 写独立目录、日志、结果和 manifest。manifest 记录协议、fold、split 索引哈希、环境、初始化哈希、掩码哈希、每轮 test 调用次数和最终选择 epoch。调度器按完成标记续跑，并在每个 baseline/JEPA 对完成后执行配对审计。

正式启动前必须完成：官方协议单元测试、全套测试、四数据集两方法的短 smoke、manifest 配对审计。任何一步失败都不得启动 640-run 队列。

## 成功标准

- official 生命周期测试证明 test 每个 epoch 调用一次，且选择只依赖 validation。
- IEMOCAP official 的 validation/test 索引相同、训练索引与其不相交、fold 固定为 5。
- K=640 的任务清单无重复、无遗漏，GPU4 不在调度列表。
- paired audit 对初始化、数据、掩码和公共配置全部通过。
- 后台队列启动后至少完成首批任务且没有任务级错误，随后可持续续跑。
