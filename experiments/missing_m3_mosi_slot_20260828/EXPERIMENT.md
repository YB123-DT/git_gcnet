# CMU-MOSI Modality-Slot Fusion

## 研究问题

当前 Missing-M3 使用 observed modality latent 的等权均值构造 GCNet node。该操作可能在进入 GCNet 前稀释 CMU-MOSI 的强 Text 信息。本实验只将其替换为固定 A/T/V 槽位拼接，判断保留模态身份是否改善 MOSI。

## 唯一变量

- Control（继承，不重跑）：`fusion_type=mean`；
- Treatment：`fusion_type=slot`；
- Slot 公式：`[A_slot; T_slot; V_slot; pattern_embedding] -> LayerNorm -> Linear -> GELU -> Dropout`；
- 缺失槽位严格为零；
- M3 Predictor、EMA、JEPA loss、GCNet、mixed-rate schedule 与 checkpoint selection 均保持不变。

本轮不研究六方向梯度冲突，不运行 attention、`lambda_J=0` 或 Original。

## 协议

- Dataset：CMU-MOSI official split，fold 1；
- Features：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`；
- Seeds：66、67、68、69、70；
- 每个 seed 训练一个模型，batch 均衡轮换 missing rates 0.0–0.7；
- 八 rate validation W-F1 等权均值选择一个 checkpoint；
- 同一 checkpoint 测试全部八个 rates；
- GPU：0、1、2、3、5；GPU 4 不使用；
- Epochs：100。

## 实现验证

- 新测试经历预期红灯：4 tests 因 `fusion_type` 参数不存在而失败；
- 实现后：`18 passed`；
- MOSI GPU smoke：1 epoch 完成，train W-F1=0.5242、val8 W-F1=0.2544，forward/backward/EMA/checkpoint/test 均完成；
- 远程 Python：`/data2/yb/reproduction_envs/gcnet-official/bin/python`；
- 远程源码：`/data2/yb/paper/GCNet_TPAMI_single_view_dev`；
- 远程结果：`/data2/yb/remote_experiments/missing_m3_mosi_slot_20260828`。

## 正式结果

5/5 个正式任务完成。每个 seed 保存 100 epoch history、一个最佳 checkpoint 的指标和八个 missing rates 的 prediction NPZ；checkpoint 保留在 biggpu，不进入 Git。

| Miss | S66 | S67 | S68 | S69 | S70 | Slot W-F1 Mean ± SD | Mean control | Delta | 正向 seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.69 | 84.76 | 85.63 | 86.20 | 85.24 | 85.50 ± 0.48 | 85.44 | +0.07 | 3/5 |
| 0.1 | 84.30 | 83.99 | 82.75 | 82.85 | 83.97 | 83.57 ± 0.64 | 82.52 | +1.06 | 4/5 |
| 0.2 | 80.51 | 81.06 | 81.81 | 81.55 | 80.58 | 81.10 ± 0.51 | 80.72 | +0.38 | 3/5 |
| 0.3 | 78.11 | 79.11 | 79.42 | 81.27 | 79.79 | 79.54 ± 1.03 | 79.06 | +0.47 | 3/5 |
| 0.4 | 76.39 | 74.40 | 77.67 | 77.29 | 78.44 | 76.84 ± 1.39 | 76.84 | -0.00 | 3/5 |
| 0.5 | 74.02 | 75.34 | 76.39 | 75.36 | 74.24 | 75.07 ± 0.86 | 74.06 | +1.00 | 3/5 |
| 0.6 | 73.01 | 75.30 | 75.12 | 75.13 | 75.14 | 74.74 ± 0.87 | 74.27 | +0.47 | 3/5 |
| 0.7 | 73.20 | 71.36 | 70.31 | 75.39 | 69.54 | 71.96 ± 2.11 | 71.29 | +0.67 | 4/5 |

| Miss | Acc-2 | W-F1 | MAE | Corr |
|---:|---:|---:|---:|---:|
| 0.0 | 85.61 | 85.50 | 0.820 | 0.773 |
| 0.1 | 83.66 | 83.57 | 0.876 | 0.734 |
| 0.2 | 81.22 | 81.10 | 0.925 | 0.701 |
| 0.3 | 79.63 | 79.54 | 0.953 | 0.679 |
| 0.4 | 77.07 | 76.84 | 1.002 | 0.636 |
| 0.5 | 75.21 | 75.07 | 1.046 | 0.604 |
| 0.6 | 74.97 | 74.74 | 1.058 | 0.594 |
| 0.7 | 72.35 | 71.96 | 1.110 | 0.549 |

最佳 epoch：S66=68、S67=54、S68=65、S69=94、S70=70。

## 配对审计

- 新旧 40/40 个 test mask SHA256 完全一致，因此 delta 是相同 seed、相同 utterance mask 的严格配对比较；
- 40/40 个 prediction NPZ 的 Acc-2、W-F1、MAE、correlation 均从原始 prediction/label 重算一致；
- 5 份 history 均为 100 epochs，所有记录为有限值；
- Slot 参数量 32,089,733，较 mean 增加 198,144 个参数；
- 五个任务于 13:39:55–13:39:56 UTC 启动，于 13:42:03–13:42:10 UTC 完成，单任务约 2 分 7 秒至 2 分 15 秒。

远程 checkpoint SHA256：

| Seed | SHA256 |
|---:|---|
| 66 | `fa709914a3fe63ae6ee8a1b991cdd2a358889165aba67cee6680248002c622dd` |
| 67 | `a28fb11c931c590b6f0626f871ae1e66821c2558e3adc6c1f375b26acd7e622a` |
| 68 | `3de9c2f8815eabd82e17a7570ea38e236d10618bcd594645f1e7b2d5524256cb` |
| 69 | `43fd0d5c6efa3aa8c0495c367086a0aed641d54729801bc925235e54e15fe8ef` |
| 70 | `4ff51e1d9aede1b6255dfbbc9cd5e9352505b0706800bded790da3685bbc354c` |

## 结论

Slot Fusion 在八个 rates 中七个取得正均值 delta，八 rate 平均提升 `+0.51` W-F1，最大提升出现在 miss 0.1（`+1.06`）和 miss 0.5（`+1.00`），且每个 rate 的跨 seed 标准差均降低。结果说明 slot 变体温和优于 mean；保留模态身份是合理解释，但 Slot 同时增加 198,144 个参数，尚未运行 parameter-matched control，因此容量效应不能排除。

但该改善不是所有 seed 一致：40 个配对单元中 26 个为正；按每个 seed 的八 rate 平均，S67/S69/S70 为正，S66/S68 为负。miss 0 的五种子均值仅 85.50，距离 88 的目标仍约 2.50。因此本实验是温和正向，不足以声称 MOSI 已被解决，也不能把提升归因于六方向梯度协同。
