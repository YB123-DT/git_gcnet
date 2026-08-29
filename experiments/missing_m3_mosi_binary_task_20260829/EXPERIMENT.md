# CMU-MOSI Binary Task Alignment（seed 66）

## 目的

判断 Missing-M3 在 CMU-MOSI 上低于部分二分类方法，是否主要来自 GCNet 原始的回归训练目标。该实验只替换监督任务：

- Regression control：单输出、全部有效标签参与 MSE，测试时用零阈值计算 Acc-2/W-F1；
- Binary：双输出、负标签映射为 0、正标签映射为 1，`y=0` 不参与 CE 和二分类指标；
- `y=0` utterance 仍保留在 Temporal/Speaker graph 和 JEPA 路径中；
- feature、mask、graph、JEPA、optimizer、100 epochs、八 rate 同批训练及 validation selection 全部不变；
- Original/Regression 结果直接继承，没有重新训练。

这是一项 task-protocol alignment 诊断，不是模型结构贡献。

## 正式协议

- Dataset：CMU-MOSI；seed 66；fold 1；Slot fusion；hidden 200；window past/future 2/2；
- Frozen features：wav2vec-large-c-UTT（512D）、DeBERTa-large-4-UTT（1024D）、MANet-UTT（1024D）；
- 一个模型在每个 source batch 中覆盖 missing rates 0.0--0.7；
- 最佳 checkpoint 仅由八个 validation W-F1 的均值选择；
- 预注册 gate：miss0 至少 87.5、相对 regression 至少 +1.0，且非零 rate 均值 delta 至少 -0.5。

## 结果

Binary 最佳 epoch 为 48，validation 八-rate 平均 W-F1 为 77.15%。

| Missing rate | Binary W-F1 | Regression W-F1 | Delta |
|---:|---:|---:|---:|
| 0.0 | 85.80 | 86.56 | -0.77 |
| 0.1 | 84.45 | 84.30 | +0.15 |
| 0.2 | 80.72 | 81.34 | -0.62 |
| 0.3 | 79.06 | 81.02 | -1.96 |
| 0.4 | 77.11 | 78.59 | -1.48 |
| 0.5 | 72.88 | 75.10 | -2.22 |
| 0.6 | 71.81 | 75.11 | -3.30 |
| 0.7 | 71.82 | 72.69 | -0.87 |
| Nonzero mean | 76.84 | 78.31 | -1.47 |

Gate 结论：**FAIL**。miss0 的绝对值、miss0 delta 和 nonzero mean delta 三项均未通过，因此不扩展 seeds 67--70。

## 审计

- history 恰好 100 epochs；EMA update 200 次；
- 8/8 prediction NPZ 已从数组重算，结果与 `metrics.json` 一致；
- 每个 rate 有 656 个 nonzero test utterances，`continuous_labels` 中零值数量为 0；
- 八个 rate 均同时预测两个类别，无单类别坍塌；
- 8/8 full-valid mask SHA256 与 paired regression control 完全一致；
- Binary 参数量 32,090,234，Regression 参数量 32,089,733，差值 501，恰好是分类头新增的一行权重与一个 bias；
- Regression、IEMOCAP 和 MOSEI 的旧 prediction NPZ key 集保持不变；Binary 额外保存原连续标签用于审计。

## 结论

在严格配对条件下，纯二分类 CE 没有弥补 MOSI 差距，反而降低 miss0 和非零 missing-rate 总体表现。因此当前不足不能归因于“GCNet 使用回归 MSE 而其他方法使用二分类 CE”这一单一因素。后续不继续调 CE 权重、类别权重或阈值；按预注册路线，下一项应检查 MOSI 单说话人条件下 Speaker branch 的退化与冗余。

