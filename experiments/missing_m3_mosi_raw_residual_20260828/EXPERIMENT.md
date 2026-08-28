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

## 正式状态

五种子正式实验尚未汇总。

