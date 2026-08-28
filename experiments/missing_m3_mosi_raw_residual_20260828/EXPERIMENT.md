# CMU-MOSI Raw-Residual Missing-M3

## 目标

取消现有 Missing-M3 在分类路径中的 2560D→256D 提前压缩。保留官方不完整 raw feature blocks，并通过零初始化 modality-specific adapter 将 Student latent 以 residual 方式接入分类路径。

Primary gate：CMU-MOSI `miss=0` 五种子平均 W-F1 ≥ 88.0。

## 唯一变量

- Existing controls（继承）：`fusion_type=mean`、`fusion_type=slot`；
- Treatment：`fusion_type=raw-residual`；
- 原始 observed block 精确保留；
- missing/padding block 精确为零；
- M3 predictor、EMA teacher、JEPA loss、GCNet、mixed-rate schedule 与 checkpoint selection 不变。

本轮不运行 Original/mean/slot，不增加 reconstruction、pattern attention、rate embedding 或 gradient-conflict optimizer。

## 协议

- Dataset：CMU-MOSI official split，fold 1；
- Features：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`；
- Seeds：66、67、68、69、70；
- 每 seed 一个 mixed-rate model，batch 均衡轮换 missing rates 0.0–0.7；
- 八 rate validation W-F1 等权均值选择一个 checkpoint；
- 同一 checkpoint 测试全部八个 rates；
- Epochs：100；GPU：0、1、2、3、5；禁用 GPU 4。

## 实现验证

- TDD 红灯：encoder import 不存在；模型路由拒绝 `raw-residual`；
- TDD 绿灯：22 tests passed；
- 零初始化 output 与 `features * expanded_availability` 精确相等；
- 七 pattern、missing-value leakage、padding、recurrent width 测试通过；
- gradient 到达 Student Projector、adapter、GCNet 与 predictor；
- CMU-MOSI 1-epoch GPU smoke 完成：train W-F1=0.5170、val8 W-F1=0.2544、classification loss=2.2884、JEPA loss=1.1638；
- Smoke parameter count=36,168,325；trainable=35,308,165；
- 远程源码：`/data2/yb/paper/GCNet_TPAMI_single_view_dev`；
- 远程结果：`/data2/yb/remote_experiments/missing_m3_mosi_raw_residual_20260828`。

## 正式结果

5/5 个任务完成。GPU 0–3 在启动时已被其他任务占用，因此未完成的 S66–S69 被立即停止并清理，随后在空闲 GPU 5/6/7 分两批从头运行；没有混用 partial checkpoint。

| Miss | S66 | S67 | S68 | S69 | S70 | Raw W-F1 Mean ± SD | Slot control | Delta | 正向 seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.61 | 85.52 | 85.17 | 85.72 | 84.68 | 85.34 ± 0.38 | 85.50 | -0.16 | 1/5 |
| 0.1 | 84.07 | 83.42 | 81.75 | 82.46 | 82.74 | 82.89 ± 0.80 | 83.57 | -0.69 | 0/5 |
| 0.2 | 81.55 | 81.49 | 80.69 | 81.11 | 78.87 | 80.74 ± 0.99 | 81.10 | -0.36 | 2/5 |
| 0.3 | 80.15 | 80.05 | 78.73 | 79.41 | 78.35 | 79.34 ± 0.71 | 79.54 | -0.20 | 2/5 |
| 0.4 | 76.09 | 73.79 | 75.90 | 77.47 | 78.51 | 76.35 ± 1.60 | 76.84 | -0.49 | 2/5 |
| 0.5 | 75.62 | 75.35 | 73.91 | 72.93 | 73.21 | 74.20 ± 1.10 | 75.07 | -0.86 | 2/5 |
| 0.6 | 76.10 | 75.49 | 73.18 | 72.13 | 73.91 | 74.16 ± 1.46 | 74.74 | -0.58 | 2/5 |
| 0.7 | 74.61 | 72.39 | 68.94 | 72.21 | 73.76 | 72.38 ± 1.94 | 71.96 | +0.43 | 3/5 |

| Miss | Acc-2 | W-F1 | MAE | Corr |
|---:|---:|---:|---:|---:|
| 0.0 | 85.43 | 85.34 | 0.796 | 0.779 |
| 0.1 | 82.93 | 82.89 | 0.856 | 0.739 |
| 0.2 | 80.79 | 80.74 | 0.904 | 0.706 |
| 0.3 | 79.39 | 79.34 | 0.946 | 0.676 |
| 0.4 | 76.37 | 76.35 | 0.992 | 0.639 |
| 0.5 | 74.30 | 74.20 | 1.041 | 0.600 |
| 0.6 | 74.18 | 74.16 | 1.051 | 0.593 |
| 0.7 | 72.32 | 72.38 | 1.096 | 0.544 |

最佳 epoch：S66=63、S67=39、S68=70、S69=56、S70=86。

## 完整性与 provenance

- 40/40 prediction NPZ 指标重算一致；
- 40/40 test mask SHA256 与 Slot 同 seed-rate 完全一致；
- 5×100 history 均完整且有限；
- parameter count=36,168,325，较 Slot 增加 4,078,592；
- 每个正式任务约 2 分 5 秒至 2 分 17 秒。

| Seed | Remote checkpoint SHA256 |
|---:|---|
| 66 | `5ba7cc772293c78ccf69dc05c671554876bef45439e88e1119affef16cb92aa2` |
| 67 | `8148636477719b0c2c7165340acbe074e000b31c79cc8775f0311cbc98876ce2` |
| 68 | `d524a57319ed3413c61747d2df635e0240a628510e3abd6df00a7723a3c7af15` |
| 69 | `649b2937a0078154d219ab3c003bb3a324746d03420da77b695848f9ea169f28` |
| 70 | `30e2b78b4f6efcb3a09180501191285a783ba5a1a1cf3dc838c35a7c1abb1614` |

## 门槛判定

- Primary：miss=0 mean 85.34 < 88.00，FAIL；
- Secondary：nonzero-rate mean 比 Slot 低 0.39，不超过 0.5 容忍线，PASS；
- Seed gate：仅 1/5 miss=0 seeds 超过 Slot，FAIL。

Raw-Residual 未解决 MOSI W-F1。虽然 miss=0 的 MAE/correlation 从 Slot 的 0.820/0.773 改善到 0.796/0.779，但阈值为零的二分类 W-F1 下降。这说明 raw information bottleneck 不是主要限制；下一轮应检查 MOSI 目标对齐和官方 `time-attn` 配置，而不是继续扩大输入表示。

