# Single-View Missing-M3 GCNet — IEMOCAPSix

## 问题

检验 `Observed-Set Fusion → GCNet → Emotion Head` 是否能通过训练期 M3/EMA 缺失 latent 预测，学习一个可跨八个缺失率使用的统一模型。

## 协议

- 数据集：IEMOCAPSix。
- Fold：5。
- Seeds：66、67、68、69、70。
- Train：每个 seed 一个模型，batch 均衡轮换 `0.0–0.7` 八个 missing rates。
- Validation：每个 epoch 在八套固定 mask 上评估，以八 rate 等权平均 W-F1 选 checkpoint。
- Test：同一个最佳 checkpoint 测试八套固定 mask。
- 特征：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`。
- GCNet：LSTM、`windowp=2`、`windowf=2`、`hidden=200`、`dropout=0.5`。
- M3：latent 256、4 experts、Top-2、`lambda_J=0.1`、temperature 0.03。
- EMA：tau 0.996，optimizer step 后更新。
- Original reconstruction：关闭。
- Inference predictor/teacher：关闭。
- Original：继承，不重新训练。

## 验证证据

- 定向单元测试：biggpu `s0` 环境，8/8 通过。
- Official GCNet 环境单 batch，`eta=0.5`：classification loss 1.7902、JEPA loss 3.8293、missing targets 1969、peak GPU 746.25 MiB；forward/backward/optimizer/EMA 均完成。

## 正式输出

每个 seed 目录必须包含：

- `config.json`
- `history.json`
- `best.pt`
- `metrics.json`
- `predictions_miss_0p0.npz` 至 `predictions_miss_0p7.npz`

最终汇总按 missing rate 报告五 seeds 的 W-F1、Macro-F1、Accuracy 均值与标准差。
