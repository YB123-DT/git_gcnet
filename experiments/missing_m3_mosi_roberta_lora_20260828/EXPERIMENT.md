# CMU-MOSI RoBERTa-Large LoRA 文本特征

## 唯一变量

冻结本地 RoBERTa-large 主干，只在 24 层 attention query/value 注入 LoRA
（48 targets，rank 8，alpha 16），用 MOSI train labels 和 SmoothL1 适配；validation
MAE 选择 checkpoint，test labels 不参与训练或选模。最后一层非特殊 token 均值形成
1024D UTT feature，替换 DeBERTa；下游 Slot Missing-M3、all-rates 和其他配置不变。

## 上游结果与审计

- 最佳 epoch：13；early stop 后共 17 epochs；
- validation MAE：0.6529；correlation：0.8241；W-F1：86.05%；
- 48 个 LoRA target，约 78.95 万个可训练参数；
- 2,199 个 feature，split 为 1284/229/686；
- 所有 feature 均为 `[1024]`、float32、finite；
- feature aggregate SHA256：`0f4c45e6d9249404367a6e533d4248d00ca42503513918d33266cef574461653`；
- adapter SHA256：`e7b068b3fee3300b195fac955e6db4b5ba100d84ddade47e5a413335f68082f2`。

## 下游 seed 66

| Miss | RoBERTa-LoRA | DeBERTa control | Delta |
|---:|---:|---:|---:|
| 0.0 | 86.23 | 86.56 | -0.34 |
| 0.1 | 84.50 | 84.30 | +0.19 |
| 0.2 | 82.13 | 81.34 | +0.79 |
| 0.3 | 80.02 | 81.02 | -1.00 |
| 0.4 | 77.75 | 78.59 | -0.84 |
| 0.5 | 77.05 | 75.10 | +1.96 |
| 0.6 | 74.96 | 75.11 | -0.15 |
| 0.7 | 74.46 | 72.69 | +1.78 |

nonzero-rate mean 为 78.70，较 control 提高 0.39；miss=0 未达到 87.0 gate，
所以该单独替换路线 FAIL，不因后续 ensemble 结果追溯性地修改门槛。

值得注意的是，miss=0 MAE 从 0.858 降到 0.702，correlation 从 0.775 升到
0.823，但零阈值 W-F1 略降。这说明新文本特征改善连续 sentiment ordering，却没有
单独解决二分类决策边界；它与 DeBERTa 的互补性需另做锁定实验。

## 验证

- remote unit/regression tests：41 passed；
- 真实 RoBERTa/PEFT 前后向：1024D、48 targets、梯度 finite；
- 8/8 masks 与 control 相同；
- 8/8 prediction NPZ 的 accuracy/W-F1 独立重算一致；
- best checkpoint epoch 24，validation eight-rate mean W-F1 82.31%；
- 下游参数量 32,089,733，与 DeBERTa control 相同。
