# Single-View Missing-M3 GCNet — IEMOCAPSix

## 问题

检验 `Observed-Set Fusion → GCNet → Emotion Head` 是否能通过训练期 M3/EMA 缺失 latent 预测，学习一个可跨八个缺失率使用的统一模型。

## 协议

- 数据集：IEMOCAPSix。
- Fold：5。
- Seeds：66、67、68、69、70。
- Train：每个 seed 一个模型，batch 均衡轮换 `0.0–0.7` 八个 missing rates。
- Validation：每个 epoch 在八套固定 mask 上评估，以八 rate 等权平均 W-F1 选 checkpoint。
- Test：同一个最佳 checkpoint 测试八套固定 mask。
- 特征：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`。
- GCNet：LSTM、`windowp=2`、`windowf=2`、`hidden=200`、`dropout=0.5`。
- M3：latent 256、4 experts、Top-2、`lambda_J=0.1`、temperature 0.03。
- EMA：tau 0.996，optimizer step 后更新。
- Original reconstruction：关闭。
- Inference predictor/teacher：关闭。
- Original：继承，不重新训练。

## 验证证据

- 定向单元测试：biggpu `s0` 环境，8/8 通过。
- Official GCNet 环境单 batch，`eta=0.5`：classification loss 1.7902、JEPA loss 3.8293、missing targets 1969、peak GPU 746.25 MiB；forward/backward/optimizer/EMA 均完成。

## 正式输出

每个 seed 目录必须包含：

- `config.json`
- `history.json`
- `best.pt`
- `metrics.json`
- `predictions_miss_0p0.npz` 至 `predictions_miss_0p7.npz`

最终汇总按 missing rate 报告五 seeds 的 W-F1、Macro-F1、Accuracy 均值与标准差。

## 正式结果

五个 seed 均完成 100 epoch。按八 rate validation W-F1 等权平均选择的最佳 epoch：

| Seed | Best epoch | Validation-8 mean W-F1 |
|---:|---:|---:|
| 66 | 89 | 61.86 |
| 67 | 88 | 61.87 |
| 68 | 100 | 61.65 |
| 69 | 84 | 63.08 |
| 70 | 97 | 63.18 |

同一 checkpoint 测试八个 rate 的 W-F1：

| Miss | Seed 66 | Seed 67 | Seed 68 | Seed 69 | Seed 70 | Mean ± SD |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 64.47 | 64.36 | 63.64 | 66.10 | 66.47 | 65.01 ± 1.22 |
| 0.1 | 65.86 | 64.28 | 62.36 | 65.65 | 66.12 | 64.85 ± 1.56 |
| 0.2 | 63.65 | 62.19 | 63.74 | 66.43 | 65.07 | 64.22 ± 1.60 |
| 0.3 | 62.18 | 64.56 | 60.87 | 64.60 | 64.70 | 63.38 ± 1.76 |
| 0.4 | 63.05 | 62.06 | 60.44 | 62.83 | 63.77 | 62.43 ± 1.27 |
| 0.5 | 61.20 | 59.66 | 61.44 | 61.49 | 62.25 | 61.21 ± 0.95 |
| 0.6 | 61.38 | 60.40 | 58.63 | 60.99 | 60.60 | 60.40 ± 1.06 |
| 0.7 | 60.05 | 57.76 | 59.53 | 60.35 | 57.53 | 59.04 ± 1.31 |

完整指标：

| Miss | W-F1 | Macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 0.0 | 65.01 ± 1.22 | 63.68 ± 1.27 | 65.08 ± 1.32 |
| 0.1 | 64.85 ± 1.56 | 63.84 ± 1.82 | 64.84 ± 1.54 |
| 0.2 | 64.22 ± 1.60 | 63.31 ± 1.48 | 64.18 ± 1.52 |
| 0.3 | 63.38 ± 1.76 | 62.32 ± 2.20 | 63.35 ± 1.64 |
| 0.4 | 62.43 ± 1.27 | 61.57 ± 1.56 | 62.29 ± 1.21 |
| 0.5 | 61.21 ± 0.95 | 60.21 ± 1.19 | 61.23 ± 0.93 |
| 0.6 | 60.40 ± 1.06 | 59.74 ± 1.93 | 60.26 ± 0.97 |
| 0.7 | 59.04 ± 1.31 | 58.21 ± 1.54 | 58.94 ± 1.36 |

### 历史 Original 参考

下表 Original 来自此前每个 rate 独立训练的 fold-5 五 seed 结果。它与本实验使用不同 mask 生成协议，因此是 historical reference，不是严格 paired A/B。

| Miss | Missing-M3 | Historical Original | Difference |
|---:|---:|---:|---:|
| 0.0 | 65.01 | 62.04 | +2.97 |
| 0.1 | 64.85 | 62.54 | +2.31 |
| 0.2 | 64.22 | 60.20 | +4.02 |
| 0.3 | 63.38 | 59.34 | +4.04 |
| 0.4 | 62.43 | 58.22 | +4.21 |
| 0.5 | 61.21 | 57.99 | +3.22 |
| 0.6 | 60.40 | 55.30 | +5.10 |
| 0.7 | 59.04 | 55.84 | +3.20 |

八 rate 平均 W-F1 为 62.57；Historical Original 为 58.93，差值 +3.63 points。该差值同时包含 observed-set node construction、mixed-rate training 与 M3 objective 的影响，不能单独归因于 M3。

## 完整性检查

- 每个 seed：1 个 best checkpoint、100 条 epoch history、8 个 prediction NPZ。
- 每个 rate：1623 条 labels/predictions，六个类别均有输出。
- NPZ 重算的 W-F1、Macro-F1、Accuracy 与 `metrics.json` 完全一致。
- 每个 seed 的八个 test mask SHA256 均不相同。
- 无 NaN、无进程失败、无单类坍塌。
