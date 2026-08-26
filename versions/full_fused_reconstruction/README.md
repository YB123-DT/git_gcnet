# Full-Fused Reconstruction

这个完成版本不增加模型参数，也不把重建结果送回分类器。它只改变原
`linear_rec` 的监督选择：当一个真实 utterance 至少缺失一个模态时，对完整
Audio/Text/Visual 三个目标都计算按模态平衡的 MSE；完整 utterance 和 padding
不参与重建损失。

```text
masked A/T/V -> Original GCNet -> linear_rec -> concatenated A/T/V prediction
                                      |
                                      +-> complete A/T/V target for incomplete utterances
```

锁定条件：Original RGCN、`hidden1 + hidden2`、`reccls_flag=false`、原分类器和
训练协议不变。正式结果覆盖 IEMOCAPSix 与 CMU-MOSI，各 8 missing rates ×
10 seeds。
