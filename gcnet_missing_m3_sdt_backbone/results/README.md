# 正式结果目录

**当前状态：`PENDING`。** Seeds 66–70 的正式训练尚未完成，本目录目前没有可报告的
treatment 分数。

## 目录结构

正式 runner 默认写入：

```text
results/
└── formal/
    ├── manifest.json
    ├── runner_status.json
    ├── seed_66/
    ├── seed_67/
    ├── seed_68/
    ├── seed_69/
    └── seed_70/
```

每个 seed 的可归档小型文件为：

```text
config.json
history.json
metrics.json
status.json
train.log
```

`best.pt` 和 `predictions_miss_*.npz` 只属于运行期产物。runner 会在确认结果完整后删除
它们，GitHub 也不得提交 checkpoint、大型 NPZ、数据集或特征文件。

## 完成判定

一个 seed 只有同时满足以下条件才标记为完成：

- `config.json` 的全部字段与锁定的 `SDTTrainConfig` 一致，而不只检查 seed、学习率等
  少数字段；
- `history.json` 恰好包含连续的 100 个 epoch；
- `metrics.json` 的 backbone 为 `sdt-style-full-context`；
- 正式 test 模式包含 `0.0` 至 `0.7` 的 8 个 rate；
- 每个 rate 都包含有限 W-F1、prediction std、sign count 和匹配的 mask SHA；
- `train.log` 存在。

`manifest.json` 每次均从这些持久文件重新检查，不能单独作为完成证据。
Validation-only 结果不会被正式 runner 标记为完成；pending seed 重跑前会清除旧
config/history/metrics/log，防止不同尝试的文件被拼成伪完整结果。

## 正式汇总字段

完成后，结果摘要至少报告：

- 逐 seed 的最佳 epoch；
- 逐 seed validation 8-rate mean W-F1；
- 逐 seed、逐 missing rate 的 test W-F1；
- 五种子 validation 8-rate mean、high-missing (`0.4`–`0.7`) mean 和 miss-0；
- 相对 inherited strict control 的逐 seed delta 与正向 seed 数；
- 8 个 test mask 的 SHA256；
- 模型总注册/可训练参数与主干 registered/active 参数；
- commit、环境和配置 provenance；
- non-collapse 检查结果。

## 预注册判据

| 指标 | Control | Candidate 门槛 |
| --- | ---: | ---: |
| Validation 8-rate mean W-F1 | 78.7675 | ≥ 79.2675 |
| High-missing validation mean W-F1 | 74.9589 | ≥ 74.9589 |
| Miss-0 validation W-F1 | 85.6461 | ≥ 85.3461 |
| 正向 seed 数 | — | ≥ 4/5 |

还必须无单一符号输出、常量输出、非有限 loss 或表示异常坍塌。Test 结果不参与门槛选择，
也不能用于调整当前候选。

## 资源画像不是正式结果

上级目录的 [`PROFILE.json`](../PROFILE.json) 只记录 1 个真实 CMU-MOSI batch 的
前向、反向、optimizer/EMA 更新、耗时和显存。Candidate 峰值已分配显存为
0.5352 GiB，因此 runner 使用 `jobs_per_gpu=2`；该文件不含可用于论文比较的分数。

Original/control 结果从既有严格复现继承，本目录不会生成 Original 训练任务。
