# CMU-MOSI Missing-M3 JEPA Loss 严格消融

## 研究问题

当前 Slot Missing-M3 同时包含两条训练信号：情绪回归与训练期 missing-latent JEPA。
此前多数据集实验尚未运行 `lambda_J=0` 控制，因此不能确定 MOSI 的效果来自 Slot Online
Encoder，还是来自 EMA Teacher、MMoE Predictor 和 JEPA target prediction。

本实验只将：

```text
lambda_J: 0.1 -> 0.0
```

模型、参数、初始化、Predictor/Teacher forward、EMA update、dropout RNG、Slot、GCNet、mask、
optimizer、学习率和 checkpoint selection 均不改变。`lambda_J=0` 时仍计算并记录 JEPA loss，
但该项不向模型提供梯度。因此这是纯 loss contribution ablation，不是删除分支后顺便改变
训练随机过程。

当前模型没有 Original raw-feature reconstruction。两组训练目标分别是：

```text
With JEPA: 情绪回归 Loss + 0.1 * Missing-Latent JEPA Loss
No JEPA:   情绪回归 Loss
```

## 锁定协议

- 数据集：CMU-MOSI，fold 1；
- frozen features：wav2vec-large-c、DeBERTa-large-4、MANet；
- 一个 all-rates-per-batch checkpoint 测试 missing rate `0.0,...,0.7`；
- seeds 66--70，100 epochs，batch size 32；
- Slot、Regression-MSE、DualGate MMoE、hidden 200、latent 256；
- window past/future 均为 2，time attention 关闭；
- Adam learning rate `5e-4`；
- validation 八-rate W-F1 均值选择 checkpoint；
- `lambda_J=0.1` 五种子结果直接继承，不重新训练；
- PCIR、Local Context、Track 和 test-time completion 全部关闭。

JEPA contribution 定义为：

```text
With-JEPA W-F1 - No-JEPA W-F1
```

有效门槛：八-rate 和 high-missing contribution 均为正，并且两者均至少 3/5 seeds 为正。

## 五种子逐 missing-rate Test W-F1

| Miss | No JEPA | With JEPA | JEPA contribution (pp) | 正向 seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 85.443 | 85.616 | +0.174 | 3/5 |
| 0.1 | 83.477 | 83.544 | +0.067 | 3/5 |
| 0.2 | 80.931 | 81.085 | +0.153 | 3/5 |
| 0.3 | 79.601 | 80.216 | +0.615 | 4/5 |
| 0.4 | 76.256 | 77.393 | +1.137 | 3/5 |
| 0.5 | 75.491 | 75.310 | -0.181 | 2/5 |
| 0.6 | 73.690 | 74.907 | +1.217 | 4/5 |
| 0.7 | 72.409 | 72.873 | +0.464 | 3/5 |

No-JEPA 的逐 rate sample standard deviation 为：

```text
0.664, 1.026, 0.962, 1.025, 1.721, 1.348, 1.834, 2.656
```

## 逐 seed 配对结果

| Seed | No-JEPA best epoch | No-JEPA Val8 | 八-rate contribution | High contribution | Miss0 contribution |
|---:|---:|---:|---:|---:|---:|
| 66 | 54 | 77.730 | +1.831 | +2.306 | +0.802 |
| 67 | 36 | 77.912 | +0.638 | +0.412 | +0.841 |
| 68 | 48 | 77.218 | +0.123 | +0.378 | -0.093 |
| 69 | 41 | 77.582 | +0.595 | +0.522 | +0.156 |
| 70 | 44 | 77.660 | -0.908 | -0.321 | -0.836 |

## 聚合与门槛

| 聚合 | No JEPA | With JEPA | Contribution | 正向 seeds | 门槛 |
|---|---:|---:|---:|---:|---|
| 八-rate mean | 78.4121 | 78.8680 | +0.4559 | 4/5 | PASS |
| High `0.4--0.7` | 74.4614 | 75.1207 | +0.6593 | 4/5 | PASS |
| Miss0 | 85.4425 | 85.6163 | +0.1738 | 3/5 | 辅助证据 |

总体 contribution 的 seed sample standard deviation 为 0.989 个百分点，说明小规模 MOSI
仍有明显随机性；但总体与高缺失方向均有 4/5 seeds 支持，因此不能把平均提升解释为单个
seed 偶然值。

## W-F1、MAE 与 correlation 的差异

JEPA 的八-rate平均影响：

```text
W-F1:       +0.456 percentage points
Correlation:+0.00385
MAE:        +0.00908  (数值越小越好，因此略有变差)
```

JEPA 在 7/8 rates 提高 W-F1，并在全部 8 rates 提高 correlation，但除 miss0.7 外均使 MAE
略高。这说明当前 target-latent prediction 更主要地塑造了情感方向和排序结构，而不是提高
连续 sentiment score 的绝对标定精度。由于正式 checkpoint/test 指标是 W-F1，该辅助目标
对当前任务仍属于有效贡献，但不能声称它同时改善了所有回归指标。

## 过拟合观察

No-JEPA 在后期的 train W-F1 达到约 0.94--0.95，而 validation 在中期达到峰值后下降；
best epochs 为 36--54。这是 MOSI 小训练集上的过拟合，不是类别坍塌。With-JEPA 同样不能
完全消除过拟合，但其总体和高缺失泛化结果更好，说明 JEPA 提供了温和而有效的辅助约束。

## 完整性审计

- 5/5 No-JEPA 任务完成 100 epochs；
- 40/40 prediction NPZ 独立重算 W-F1，与 metrics 完全一致；
- 40/40 NPZ availability SHA 与本次 metrics 一致；
- 40/40 No-JEPA mask SHA 与相同 seed/rate 的 With-JEPA 结果一致；
- 5/5 config 记录 `jepa_weight=0.0`、`node_interaction_residual=false`；
- 两组参数量均为 32,089,733，没有容量差异；
- checkpoint 只按 validation 选择，test 未参与选择。

## 结论

`COMPLETE — PASS`。

Missing-latent JEPA/MMoE training objective 对当前 MOSI Slot + GCNet 模型有可验证贡献：
八-rate W-F1 提高 0.456 个百分点，高缺失提高 0.659 个百分点，均有 4/5 seeds 支持。
因此不应删除 EMA Teacher、Missing-Latent Predictor 或 MMoE；它们不是推理模块，但训练期
确实改善了不完整输入下的泛化。

当前证据也表明 `lambda_J=0.1` 可能不是最佳权重。下一步若继续优化 MOSI，应只做
validation-only 的小范围 `lambda_J` 校准，而不是重新修改 Slot、GCNet 或增加新的融合模块。

结果位置：

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_jepa_ablation_20260829/formal/lambda_0`；
- Local：本实验目录的 `results/formal/lambda_0`；
- checkpoint 仅保留在 biggpu，不提交 Git。
