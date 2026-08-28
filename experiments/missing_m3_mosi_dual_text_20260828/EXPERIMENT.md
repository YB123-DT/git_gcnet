# CMU-MOSI 双文本单模型

## 唯一变量

把同一 utterance 的 DeBERTa 1024D 与 RoBERTa-LoRA 1024D 拼接为一个 2048D
Text modality。Text 缺失时整个 2048D block 同时置零。其余 Slot Missing-M3、
all-rates protocol、GCNet、loss 和优化器不变；推理仍为一个模型，不做 logit
ensemble。

该设计来自固定双模型 ensemble 的稳定互补证据。两个 Encoder 坐标系不对齐，
因此不做原始向量相加，而让现有 Text Student projector 从拼接表示中学习压缩。

## seed 66 结果

| Miss | Dual-text | DeBERTa control | Delta |
|---:|---:|---:|---:|
| 0.0 | 86.86 | 86.56 | +0.30 |
| 0.1 | 85.47 | 84.30 | +1.16 |
| 0.2 | 80.40 | 81.34 | -0.95 |
| 0.3 | 78.54 | 81.02 | -2.48 |
| 0.4 | 78.14 | 78.59 | -0.45 |
| 0.5 | 75.00 | 75.10 | -0.10 |
| 0.6 | 75.76 | 75.11 | +0.65 |
| 0.7 | 72.72 | 72.69 | +0.04 |

- miss=0 gate：86.86 < 88.00，FAIL；
- nonzero-rate mean：78.00 < 79.00，FAIL；
- best epoch：32；validation eight-rate mean W-F1：79.96%；
- parameter count：33,846,917，比 1024D control 增加 1,757,184。

因此不扩 seeds 67--70。直接拼接虽然保留了原始维度，但现有单个 Text projector
仍把两路表示立即压到同一个 256D latent，未能复现 late ensemble 的互补收益。
若继续单模型化，必须在压缩前使用两个独立 text projector，并把融合推迟到各自
形成 latent 后；不能再增加 GraphConv/RGCN 模块来解释这个问题。

## 审计

- source bank：2,199 个 2048D float32 finite arrays；
- feature aggregate SHA256：`9800656b02cee404e231226cabeb36d6c351886dc80b9fb0e83c66422715298e`；
- 8/8 prediction NPZ W-F1 独立重算一致；
- 8/8 mask SHA256 与 seed 66 DeBERTa control 一致；
- history 为完整 100 epochs，未出现 NaN 或坍塌。
