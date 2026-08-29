# CMU-MOSI Weight Decay Validation Screen

## 目的

No-JEPA 消融显示 MOSI 存在明显 train/validation gap，而 `lambda_J=0.1` 已通过贡献消融和
validation 权重筛选。本实验不修改模型，只检查更强 Adam weight decay 是否能稳定改善
分类泛化。

## 协议

- Dataset：CMU-MOSI，fold 1，seed 66；
- 正式配置保持 `lr=5e-4`、`lambda_J=0.1`、Slot、all-rates-per-batch、100 epochs；
- 新运行：`weight_decay=5e-5,1e-4,5e-4`；
- 继承 Control：`weight_decay=1e-5`；
- 只按 best validation 八-rate W-F1 选择；
- 候选需比 Control 至少高 0.3 个百分点才扩 seeds67--70；
- test 在 validation 选择锁定后才读取，不能改变扩展决定。

## Validation 筛选

| Weight decay | Best epoch | Best Val8 W-F1 | 相对 Control |
|---:|---:|---:|---:|
| **1e-5 Control** | **43** | **79.482** | **0.000** |
| 5e-5 | 47 | 79.677 | +0.195 |
| 1e-4 | 44 | 79.056 | -0.426 |
| 5e-4 | 51 | 78.547 | -0.935 |

`5e-5` 虽为最高 validation，但只提高 0.195，未达到预注册的 0.3 扩展门槛。因此保持
`1e-5` 正式设置，不运行候选的其余四个 seeds。

## 锁定后 Test 报告

| Weight decay | Miss0 W-F1 | 八-rate mean | High `0.4--0.7` |
|---:|---:|---:|---:|
| **1e-5 Control** | 85.805 | 79.241 | 75.064 |
| 5e-5 | 85.533 | 79.204 | 75.311 |
| 1e-4 | **86.184** | **79.361** | **75.290** |
| 5e-4 | 85.134 | 78.405 | 74.431 |

Test 进一步显示选择不稳定：`5e-5` 的 validation略高但 test总体略低；`1e-4` 的 test略高
但 validation低 0.426。不能根据后读到的 seed66 test 改选 `1e-4`，也不能把它描述为
正式提升。

24/24 新 prediction NPZ 的 W-F1 独立重算一致，24/24 mask SHA 与同 seed/rate Control
一致。

## 结论

`COMPLETE — KEEP 1e-5`。

现有分类平台不能由简单增大 weight decay 解决。当前正式 MOSI 配置继续使用：

```text
lr=5e-4
weight_decay=1e-5
lambda_J=0.1
```

不扩五种子，不继续细分 `2e-5/3e-5/7e-5` 等近邻值，避免围绕单 seed 噪声进行搜索。

结果位置：

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_weight_decay_screen_20260829/screen`；
- Local：本实验目录的 `results/screen`；
- checkpoint 只保留在 biggpu，不提交 Git。
