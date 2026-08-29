# CMU-MOSI Classification-Coupled Missing Latent

## 唯一变量

- Control（继承）：training-only DualGate Missing-M3；
- Treatment：训练和推理均保留 DualGate Missing-Latent Predictor；
- Predictor 的 regression target latent 经三个 target-specific LN+零初始化 Linear、tanh
  后，在实际缺失目标内平均，并作为 GCNet hidden residual 输入同一个 Emotion Head；
- EMA Teacher、完整 target 与 JEPA loss 仍只存在于训练期；
- 无第二次 GCNet、无 raw feature completion、无新 gate。

其余锁定 CMU-MOSI、Slot、Regression-MSE、all-rates-per-batch、100 epochs、seeds
66--70、hidden 200、window 2/2、time attention false。Control 五种子不重跑。

## 验证

- TDD：completion API 缺失时 collection error；实现后 focused 3/3 通过；
- 相关完整测试：80 passed；
- official 1-epoch：train W-F1=0.5167，val8 W-F1=0.2544；
- config 明确记录 `classification_completion=true`、`mmoe_variant=dual-gate`；
- 8/8 prediction NPZ 完整；参数量 32,476,769。

## 状态

`RUNNING`。远程正式根目录：

`/data2/yb/remote_experiments/missing_m3_mosi_classification_completion_20260829/formal`

