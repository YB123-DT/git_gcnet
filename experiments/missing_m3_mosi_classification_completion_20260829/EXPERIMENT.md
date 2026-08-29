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

`COMPLETE — FAIL`。远程正式根目录：

`/data2/yb/remote_experiments/missing_m3_mosi_classification_completion_20260829/formal`

## 五种子结果

| Miss | Completion | Training-only control | Delta | 正向 seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 84.95 | 85.76 | -0.81 | 1/5 |
| 0.1 | 82.68 | 83.05 | -0.37 | 1/5 |
| 0.2 | 80.03 | 80.79 | -0.76 | 2/5 |
| 0.3 | 78.83 | 79.20 | -0.37 | 3/5 |
| 0.4 | 76.00 | 76.20 | -0.20 | 2/5 |
| 0.5 | 74.21 | 74.80 | -0.59 | 2/5 |
| 0.6 | 72.19 | 73.30 | -1.12 | 1/5 |
| 0.7 | 71.76 | 71.37 | +0.40 | 4/5 |

- 八-rate 均值：77.581，对照 78.059，delta=-0.478；
- 非零-rate 均值：76.529，对照 76.959，delta=-0.430；
- 0.4--0.7 均值：73.539，对照 73.916，delta=-0.377；
- 总体正向 2/5 seeds，高缺失正向 2/5 seeds；
- 40/40 mask SHA256 配对一致；
- 新增 387,036 参数。

Miss=0.7 虽然平均提升 0.40 且 4/5 seeds 为正，但 sample SD 达到 4.44；seed68 在
该 rate 下降 3.29。其余七个 rate 中六个平均下降，无法将其解释为稳定高缺失收益。

## 结论

直接把预测 missing latent 作为 Emotion Head 的 residual 输入，会明显改变并加快训练，
但不能稳定提高测试泛化。预测误差在 seed 与 rate 之间被分类路径放大，符合 test-time
hallucination noise 的风险。CaM-HG 的 test-time generator 与 confidence fusion、恢复后的
hypergraph 是一个整体，不能只迁移“保留 Predictor”这一项。本变体保留为失败消融，
不作为主模型，也不继续追加 reliability gate 来事后挽救。
