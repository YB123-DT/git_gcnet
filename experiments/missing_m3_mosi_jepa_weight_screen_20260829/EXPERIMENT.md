# CMU-MOSI JEPA Weight Validation Screen

## 目的

`lambda_J=0` 五种子消融证明 training-only Missing-Latent JEPA 能提高 MOSI 的 W-F1，
但其贡献不足以解决当前分类平台。本实验只检查默认 `lambda_J=0.1` 是否明显失配，
不增加模型模块，也不使用 test 选择权重。

## 协议

- Dataset：CMU-MOSI，fold 1，seed 66；
- 其余配置保持正式 Slot Missing-M3：`lr=5e-4`、all-rates-per-batch、100 epochs、
  Regression-MSE、DualGate MMoE、hidden 200、latent 256、window 2/2；
- 新运行：`lambda_J=0.03, 0.05, 0.2`；
- 继承：`lambda_J=0` 与 `lambda_J=0.1`；
- 唯一选择指标：best validation 八-rate W-F1；
- 选择锁定前不读取候选 test 指标。

## Validation 筛选结果

| `lambda_J` | Best epoch | Best Val8 W-F1 | 相对 0.1 |
|---:|---:|---:|---:|
| 0 | 54 | 77.730 | -1.752 |
| 0.03 | 44 | 78.457 | -1.025 |
| 0.05 | 43 | 78.653 | -0.829 |
| **0.1** | **43** | **79.482** | **0.000** |
| 0.2 | 49 | 78.036 | -1.445 |

结果呈现合理的中间最优：完全关闭 JEPA 最差，弱权重逐步改善，过强的 0.2 又下降。
继承的 `0.1` 比第二名 `0.05` 高 0.829 个 validation W-F1 点，因此不存在需要扩展五种子
的新权重候选。

## 锁定后 Test 报告

在 validation 已锁定 `0.1` 且扩展决定完成后，读取 test 只用于报告：

| `lambda_J` | Miss0 W-F1 | 八-rate mean | High `0.4--0.7` |
|---:|---:|---:|---:|
| 0.03 | **86.416** | 79.239 | 74.944 |
| 0.05 | 84.383 | 78.109 | 74.673 |
| **0.1** | 85.805 | **79.241** | **75.064** |
| 0.2 | 85.398 | 77.711 | 72.830 |

`0.03` 在单个 seed 的 Miss0 高于 `0.1` 0.611 点，但八-rate mean 低 0.001、高缺失低
0.120，而且 validation 低 1.025。这个 test 观察不能推翻预先完成的 validation 选择，
也不能作为扩 seeds67--70 的依据。

24/24 新 prediction NPZ 的 W-F1 独立重算一致，24/24 mask SHA 与同 seed/rate 的
`lambda_J=0.1` 一致。

## 判定

`COMPLETE — KEEP 0.1`。

本轮不读取候选 test 来反向选择，也不运行 seeds67--70。当前 MOSI 分类低分不能通过继续
微调 `lambda_J` 解释；`0.1` 已获得消融贡献证据和 validation 权重筛选的双重支持。

结果位置：

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_jepa_weight_screen_20260829/screen`；
- Local：本实验目录的 `results/screen`；
- checkpoint 只保留在 biggpu，不提交 Git。
