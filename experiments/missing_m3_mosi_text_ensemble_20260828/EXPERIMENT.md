# CMU-MOSI 文本互补性确认

## 问题与锁定协议

RoBERTa-LoRA 单独替换 DeBERTa 后，seed 66 的 regression MAE/correlation 明显
改善，但零阈值 W-F1 略降。为判断两类文本证据是否互补，本实验使用 seed 66
validation 的八个 rates 选择一个全局权重，并在观察 seeds 67--70 结果前锁定：

```text
p = 0.71 * p_RoBERTa-LoRA + 0.29 * p_DeBERTa
threshold = 0
```

所有 rates 和 seeds 共用该权重。没有按 test、rate 或 seed 调参。两个组成模型均为
CMU-MOSI、Slot Missing-M3、`train_rate_mode=all`、100 epochs；seed 66 直接继承，
seeds 67--70 补齐配对训练。

## 五种子 miss=0

| Seed | DeBERTa | RoBERTa-LoRA | Fixed ensemble |
|---:|---:|---:|---:|
| 66 | 86.56 | 86.23 | **88.22** |
| 67 | 83.45 | 86.54 | **87.14** |
| 68 | 86.61 | 87.04 | **88.27** |
| 69 | 86.19 | 86.53 | **88.06** |
| 70 | 85.97 | 86.87 | **87.75** |
| Mean ± SD | 85.76 | 86.64 | **87.89 ± 0.42** |

固定 ensemble 在 5/5 seeds 中同时优于两个组成模型；但预注册的 miss=0 mean
≥88.00 门槛差 0.11，因此严格判定为 FAIL，不写成正式通过。

## 八 missing rates

| Miss | Ensemble mean ± SD | DeBERTa | RoBERTa-LoRA | Δ vs DeBERTa | Δ vs LoRA |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.89 ± 0.42 | 85.76 | 86.64 | +2.13 | +1.25 |
| 0.1 | 85.70 ± 0.22 | 83.05 | 84.88 | +2.65 | +0.82 |
| 0.2 | 83.16 ± 0.68 | 80.79 | 82.21 | +2.37 | +0.95 |
| 0.3 | 81.77 ± 1.57 | 79.20 | 80.68 | +2.57 | +1.09 |
| 0.4 | 79.40 ± 1.76 | 76.20 | 78.56 | +3.20 | +0.84 |
| 0.5 | 77.03 ± 0.76 | 74.80 | 76.93 | +2.23 | +0.10 |
| 0.6 | 76.38 ± 0.41 | 73.30 | 75.04 | +3.07 | +1.33 |
| 0.7 | 74.18 ± 1.36 | 71.37 | 73.66 | +2.82 | +0.53 |

nonzero-rate mean 为 79.66，DeBERTa 为 76.96，RoBERTa-LoRA 为 78.85。

## 完整性审计

- 40/40 DeBERTa/LoRA pairs 的 labels、availability 和 mask SHA256 一致；
- 80 个 source prediction NPZ 的 W-F1 独立重算与 metrics 完全一致；
- 40 个 ensemble NPZ 已保存；
- 8 个新增训练 history 均为 100 epochs，所有记录有限；
- GPU5/6 各并发 3 个任务，GPU7 并发 2 个任务，未使用 GPU4。

## 结论

结果支持“通用语义 DeBERTa 与 sentiment-adapted RoBERTa 表示互补”，但这里仍是
双模型推理 control，不是新的单模型主方法。下一步只验证一个由该证据直接推出的
单模型版本：把两个 1024D text views 拼接为一个 2048D Text modality，由现有
Student projector 学习融合；不修改 GCNet、JEPA 或 mask protocol。
