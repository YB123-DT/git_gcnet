# CMU-MOSI Temporal/Speaker Graph Branch 消融

## 问题

CMU-MOSI 的 `n_speakers=1`，Speaker graph 只有一种 `00` relation。本实验判断该分支是否在单说话人条件下退化为冗余或干扰。

三个条件使用相同参数量、state-dict key 和初始化顺序：

- Original `both`：`h = h_temporal + h_speaker`，五种子结果直接继承；
- `temporal-only`：只执行 Temporal graph；
- `speaker-only`：只执行 Speaker graph。

未选择的 branch module 仍被实例化，但不执行 forward、不产生梯度。实验保持 CMU-MOSI Regression-MSE、Slot、hidden 200、window 2/2、100 epochs、all-rates-per-batch、固定 mask 和 validation 八-rate checkpoint selection 不变。

## 五种子主结果

| Mode | Miss0 W-F1 | Nonzero-rate mean | 相对 Original miss0 | 相对 Original nonzero |
|---|---:|---:|---:|---:|
| Original both | **85.76** | **76.96** | — | — |
| Temporal-only | 84.83 ± 1.42 | 76.56 ± 2.61 | -0.93 | -0.40 |
| Speaker-only | 84.97 ± 1.35 | 76.53 ± 1.43 | -0.79 | -0.43 |

逐 seed：

| Seed | Original miss0 / NZ | Temporal-only miss0 / NZ | Speaker-only miss0 / NZ |
|---:|---:|---:|---:|
| 66 | 86.56 / 78.31 | 86.77 / 78.76 | 85.05 / 76.36 |
| 67 | 83.45 / 76.53 | 85.20 / 76.86 | 86.15 / 77.78 |
| 68 | 86.61 / 76.10 | 84.29 / 77.17 | 85.14 / 76.33 |
| 69 | 86.19 / 77.22 | 85.01 / 77.90 | 82.71 / 74.33 |
| 70 | 85.97 / 76.64 | 82.88 / 72.08 | 85.81 / 77.84 |

逐 missing rate 五种子均值：

| Rate | Original | Temporal-only | Delta | Speaker-only | Delta |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.76 | 84.83 | -0.93 | 84.97 | -0.79 |
| 0.1 | 83.05 | 82.20 | -0.85 | 83.04 | -0.01 |
| 0.2 | 80.79 | 80.29 | -0.50 | 80.77 | -0.03 |
| 0.3 | 79.20 | 78.91 | -0.29 | 78.62 | -0.58 |
| 0.4 | 76.20 | 75.66 | -0.54 | 75.24 | -0.96 |
| 0.5 | 74.80 | 74.00 | -0.80 | 73.54 | -1.26 |
| 0.6 | 73.30 | 73.10 | -0.21 | 72.67 | -0.63 |
| 0.7 | 71.37 | 71.74 | +0.38 | 71.82 | +0.45 |

## 审计

- 10/10 任务完成 100 epochs；
- 80/80 prediction NPZ 重算 W-F1 与 `metrics.json` 一致；
- 80/80 full-valid mask SHA256 与相同 seed Original 完全一致；
- 三种模式参数量均为 32,089,733，state-dict key 集相同；
- 单元测试证明 default/both 逐位等价，单分支不执行未选择 branch；
- 没有 checkpoint 被复制到 Git。

## 结论

“MOSI 单说话人使 Speaker branch 应被删除”的假设不成立。Temporal-only 与 Speaker-only 的五种子总体表现都低于 Original，且 Temporal-only 的 nonzero seed 方差明显增大。Speaker-only 与 Temporal-only 接近，说明单关系 Speaker graph 仍能学习有效的上下文变换；Original 的两套分支在总体上提供互补证据。

因此关闭单分支路线。该诊断也不支持把 MOSI 的约 2 点差距归因于 Speaker branch 的简单冗余。后续若研究融合，应有独立机制依据，不能把“单说话人”作为删除 Speaker branch 的理由。

