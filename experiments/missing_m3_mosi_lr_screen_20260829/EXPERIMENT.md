# CMU-MOSI Slot Missing-M3 下游 Learning Rate 筛选

## 研究问题与单变量

冻结 wav2vec、DeBERTa、MANet feature bank，只筛选当前 Slot Missing-M3 + GCNet 的
Adam learning rate。模型、Loss、all-rates-per-batch、mask、100 epochs 和验证选择协议
均不改变。

Control `lr=1e-3` 五种子直接继承。seed66 初筛新运行：

```text
3e-4, 5e-4, 2e-3
```

只能按 validation 八-rate W-F1 均值选择正式 LR；test 在选择锁定后才读取。

## Seed66 Validation 筛选

| LR | 最佳 epoch | Validation 8-rate W-F1 | 身份 |
|---:|---:|---:|---|
| 3e-4 | 49 | 78.418 | 候选 |
| **5e-4** | **43** | **79.482** | **预注册胜者** |
| 1e-3 | 44 | 77.887 | 继承 Control |
| 2e-3 | 1 | 41.874 | 坍塌 |

因此在读取 test 汇总前锁定 `5e-4`。按用户后续明确要求，`3e-4` 也补齐五种子；它属于
看到 seed66 test 后追加的 exploratory stability control，不能反过来作为预注册 LR
选择证据。

## 五种子逐 missing-rate Test W-F1

| Miss | 3e-4 | 5e-4 | 1e-3 Control | 3e-4 Δ | 5e-4 Δ |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 86.00 | 85.62 | 85.76 | +0.24 | -0.14 |
| 0.1 | 83.39 | 83.54 | 83.05 | +0.34 | +0.49 |
| 0.2 | 80.99 | 81.09 | 80.79 | +0.20 | +0.29 |
| 0.3 | 79.76 | 80.22 | 79.20 | +0.55 | +1.01 |
| 0.4 | 77.48 | 77.39 | 76.20 | +1.29 | +1.20 |
| 0.5 | 75.52 | 75.31 | 74.80 | +0.72 | +0.51 |
| 0.6 | 74.33 | 74.91 | 73.30 | +1.03 | +1.60 |
| 0.7 | 73.14 | 72.87 | 71.37 | +1.77 | +1.51 |

## 聚合结果

| LR | Miss0 | 8-rate mean | High 0.4--0.7 | 总体正向 seeds | 高缺失正向 seeds |
|---:|---:|---:|---:|---:|---:|
| 3e-4 | 86.00 | 78.83 | 75.12 | 4/5 | 5/5 |
| **5e-4** | **85.62** | **78.87** | **75.12** | **4/5** | **4/5** |
| 1e-3 | 85.76 | 78.06 | 73.92 | — | — |

相对 Control：

- `3e-4`：8-rate `+0.768`，high-missing `+1.203`，miss0 `+0.239`；
- `5e-4`：8-rate `+0.809`，high-missing `+1.204`，miss0 `-0.142`；
- `5e-4` 五个 seed 的总体 delta：`-0.099,+1.310,+1.194,+0.656,+0.984`。

正式 `5e-4` 通过预注册门槛：八-rate mean 为正、4/5 seed 为正、高缺失不下降。

## 完整性审计

- 初筛 3 个 seed66 与两组五种子扩展全部完成 100 epochs；
- 88/88 prediction NPZ 从 predictions/labels 独立重算 W-F1，与 metrics 一致；
- 88/88 mask SHA256 与相同 seed/rate 的 `1e-3` Control 一致；
- 参数量与模型 state key 不变；
- `2e-3` 从 epoch1 起保持常量预测，属于真实优化坍塌，不是文件解析错误；
- test 未参与 seed66 LR 选择。

## 结论

`COMPLETE — PASS`。

现有 `1e-3` 对 mixed-rate MOSI 下游偏高。正式后续配置改为显式 `--lr 0.0005`；不修改
全局 CLI 默认值，以免无证据影响 IEMOCAP/MOSEI。`3e-4` 是表现接近且 Miss0 更好的
探索性对照，但正式选择仍为 validation 预先锁定的 `5e-4`。

本次提升来自纯优化配置，不构成论文结构创新，但它建立了后续研究图前联合节点构造时
更可靠的 MOSI 下游基线。

## 结果位置

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_lr_screen_20260829`；
- Local：本实验目录的 `results/screen` 与 `results/formal`；
- checkpoint 只保留在 biggpu，不提交 Git。
