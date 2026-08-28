# MOSI 双文本单模型判别设计

## 依据

固定 validation 权重的 DeBERTa/RoBERTa-LoRA 双模型 ensemble 在 seeds 66--70
的 miss=0 达到 87.89±0.42，且 5/5 seeds 同时优于两个组成模型。这表明两种
文本表示包含稳定互补信息，但双模型推理不是目标架构。

## 唯一变量

对每个 MOSI utterance 构造一个新的 Text feature：

```text
T_dual = concat(T_DeBERTa[1024], T_RoBERTa-LoRA[1024])  # 2048D
```

Audio、Visual、mask、Slot Missing-M3、all-rates protocol、GCNet、loss、优化器和
checkpoint selection 全部不变。两个文本 view 共享同一个 Text availability bit；
Text 缺失时 2048D 整块同时置零。模型仍只有一个 Student/Teacher、一个 GCNet 和
一个 regression head，不进行 logit ensemble。

不做离线均值或跨空间相加，因为两个 Encoder 的坐标系没有对齐。当前 Student
projector 的输入归一化与线性投影负责从拼接表示中学习可用的统一 Text latent。

## 协议

- 数据集：CMU-MOSI；
- 第一阶段：seed 66，100 epochs，8 rates；
- 继承同 seed DeBERTa、RoBERTa-LoRA 和 ensemble 结果，不重跑；
- 所有 source UID、shape、dtype、finite 和 SHA256 在训练前审计；
- 通过门槛：miss=0 W-F1 ≥88.0，且 nonzero-rate mean ≥79.0；
- 通过后才扩 seeds 67--70；失败则关闭双文本单模型路线。

参数量增加必须单独记录，后续若作为论文结果需要 parameter-matched control；本轮
只判断互补证据能否被一个模型吸收，不把容量效应解释为结构创新。
