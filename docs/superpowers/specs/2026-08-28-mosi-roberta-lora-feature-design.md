# MOSI RoBERTa-Large LoRA 文本特征设计

## 目标

当前 frozen DeBERTa/GCNet all-rates 模型的 seed66 miss0 W-F1 为 86.56。MOSI 原始音频和视频不可用，无法做三模态端到端 Encoder adaptation；但 label pickle 保留了 2,199 条原始文本，biggpu 具有本地 RoBERTa-large 与 PEFT 环境。因此只适配主导 MOSI 的文本 Encoder，导出新的 1024D UTT 特征，再交给不变的 Missing-M3/GCNet。

## 数据边界

- Label/text：`CMUMOSI_features_raw_2way.pkl`；
- train/validation/test utterances：1284/229/686；
- 只有 train labels 参与梯度；
- validation 仅按 MAE 选择 LoRA checkpoint；
- test labels 在 LoRA 训练与选模阶段禁止读取；
- 导出阶段只读取 UID 与文本，不读取 labels。

## Encoder

- Base：`/data2/yb/pretrained/roberta-large-722cf37`；
- 24 layers，hidden size 1024；
- 冻结全部 base parameters；
- LoRA target：每层 self-attention `query` 与 `value`，共 48 个 Linear；
- rank=8，alpha=16，dropout=0.05，bias=none；
- gradient checkpointing；
- tokenizer：本地 RoBERTa tokenizer，dynamic padding，max length 192。

禁止使用 `pooler_output`，因为该 MLM checkpoint 的 pooler 不是可靠的预训练句表示。

## Pooling 与临时头

对最后一层 hidden：

```text
valid token = attention_mask AND token not in {BOS, EOS, PAD}
pooled = sum(hidden * valid) / count(valid)
```

`pooled` shape 为 `[B,1024]`。临时上游回归头：

```text
LayerNorm(1024) → Dropout(0.1) → Linear(1024,1)
```

训练 loss 仅为 SmoothL1；head 只用于让 LoRA 适配 MOSI sentiment，导出后删除。

## 优化

- seed=66；batch=16；max epochs=20；patience=3；
- AdamW；LoRA/head lr=2e-4；weight decay=0.01；
- AMP FP16；gradient clip=1.0；
- validation MAE 最小为唯一 checkpoint score；correlation/W-F1 只记录；
- 所有超参数在下游结果出现前锁定，不根据 GCNet validation/test 回调 LoRA。

## Artifact

输出目录：

```text
features/roberta-large-lora-r8-mean-UTT/
  <UID>.npy             # float32, shape [1024]
  manifest.json
  adapter/
  sentiment_head.pt
  history.json
```

manifest 至少记录：

- base model path 与 weight SHA256；
- LoRA/pooling/training config；
- best epoch 与 validation metrics；
- 2,199 UID、split UID hashes；
- 2,199 feature hashes 或一个 canonical aggregate hash；
- adapter/head hashes；
- git commit。

GCNet dataloader 直接把新目录名作为 `--text-feature`，不修改维度或读取逻辑。

## 测试与泄漏不变量

1. split UID 不交叉且覆盖 2,199；
2. 文本与 UID 严格对齐；
3. masked mean 排除 BOS/EOS/PAD；
4. base parameters frozen，只有 LoRA/head 可训练；
5. real RoBERTa integration 中 LoRA target 数为 48；
6. validation/test labels 不进入 training batch；
7. 导出 2,199 个 `[1024]` float32 finite arrays；
8. 同 seed 的 batch order 与结果可复现；
9. manifest hashes 可重算。

## 下游判别

- 使用 `train_rate_mode=all` 的 Slot Missing-M3；
- 仅把 text feature 从 DeBERTa 改为 RoBERTa-LoRA；
- MOSI seed66、100 epochs、八-rate validation selection/test；
- paired control：all-rates DeBERTa seed66，miss0=86.56；
- 扩展门槛：miss0 ≥87.0 且相对 control ≥+0.5；nonzero mean 不低于 control 超过0.5；
- 未过门槛不扩 seeds 67–70。

