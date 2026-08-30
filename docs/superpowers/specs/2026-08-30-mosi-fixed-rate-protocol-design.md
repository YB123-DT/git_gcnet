# MOSI Fixed-Rate Missing-M3 协议设计

## 目标

在模型、特征、损失、优化器和自然缺失生成方式不变时，只把当前
`all-rates-per-batch` 协议替换为 SDR-GNN 风格的“一个 missing rate 训练一个模型，
并在相同 missing rate 上选择 checkpoint 和测试”，判断统一八率训练是否压低
Slot Missing-M3 的 CMU-MOSI 表现。

## 已选方案与对照

比较过三种训练协议：

1. `all`：一个 batch 同时计算八个 rate；已有正式结果，作为统一模型对照。
2. `cyclic`：每个 batch 只计算一个 rate，但八个 rate 按 batch 轮换；不能回答固定率问题。
3. `fixed`：每个模型只接收一个预注册 rate；本实验采用。

用户已明确选择方案 3。方案 1 的结果直接继承，方案 2 不新增任务。

## 唯一变量

每个任务绑定一个 `train_missing_rate`：

```text
train η = validation-selection η = test η
```

正式矩阵为：

```text
η ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
seed ∈ {66, 67, 68, 69, 70}
总任务数 = 8 × 5 = 40
```

训练 mask 仍由现有 `ConversationMaskSchedule` 生成：请求 rate 固定，但训练期具体
utterance pattern 随既有 epoch schedule 变化。Validation/test mask 保持冻结和可追溯。

## 锁定配置

- 数据集：CMU-MOSI，官方 train/validation/test split，fold 1；
- frozen features：wav2vec-large-c、DeBERTa-large-4、MANet；
- 模型：Slot observed-set encoder + Original GCNet conversation path；
- Missing-M3：DualGate MMoE、EMA teacher、target-balanced JEPA；
- 任务：regression MSE，W-F1 为 checkpoint selection 主指标；
- hidden 200，latent 256，window past/future 2/2，time attention 关闭；
- Adam，LR `5e-4`，weight decay `1e-5`；
- JEPA weight `0.1`，100 epochs，batch size 32；
- 不启用 completion、PCIR、LoRA、conditioned readout 或新图模块。

## 训练器合同

`TrainConfig` 在末尾追加 `fixed_missing_rate: float | None = None`，以保持旧 positional
构造兼容。`train_rate_mode="fixed"` 时：

- `fixed_missing_rate` 必须精确属于八个正式 rate；
- 每个训练 batch 只构造该 rate 的一个 view；
- 每个 epoch 只在该 rate 上验证；
- 最佳 epoch 只由该 rate 的 validation W-F1 决定；
- 最佳 checkpoint 只在该 rate 上测试并只保存一个 prediction NPZ；
- metrics 显式记录训练、选模和测试 rate。

`cyclic`/`all` 时若传入 `fixed_missing_rate` 必须报错，防止 manifest 语义含混。旧模式的
默认输出和八率选择行为保持不变。

## Runner 与恢复

新增 `scripts/run_mosi_fixed_rate.py`：

- 默认生成 40 个且仅生成 40 个 Missing-M3 命令；
- GPU 只使用 0、1、2，禁止 GPU 4；
- 默认每卡最多 5 个并发任务；
- 已有任务只有在 config、100-epoch history、metrics、单-rate NPZ 均一致时才跳过；
- 不启动 Original/GCNet control；
- manifest 记录 source commit、训练命令、rate、seed、GPU、源码 SHA 和状态；
- 异常子进程被回收，其他任务继续，最终汇总失败列表。

输出目录：

```text
/data2/yb/remote_experiments/missing_m3_mosi_fixed_rate_20260830/formal/
  rate_0p0/seed_66/
  ...
  rate_0p7/seed_70/
```

## 验证与判定

不运行额外 1-epoch smoke。顺序固定为：

1. focused 单元测试验证 fixed rate 贯穿 train/validation/test；
2. `gcnet-official` 环境运行一次 CPU/构造级测试和 runner dry-run；
3. 直接启动 40 个正式 100-epoch 任务。

主报告比较：

- fixed-rate 五种子逐 rate test W-F1；
- fixed-rate 八率平均；
- 同一个 Slot Missing-M3 的既有 mixed-rate 五种子结果；
- published SDR-GNN/CaM-HG 只作协议标注后的外部参照，不作严格配对统计。

本实验回答协议效应，不把 fixed-rate specialization 声称为新模型贡献。
