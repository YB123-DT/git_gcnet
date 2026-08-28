# MOSI 文本互补性确认实验

## 研究问题

RoBERTa-LoRA 替换 DeBERTa 后，seed 66 的 miss=0 W-F1 从 86.56 降到
86.23，但 MAE 从 0.858 降到 0.702、相关系数从 0.775 升到 0.823，且
nonzero-rate mean 提高 0.39。两组配对预测的错误并不相同，因此需要判断
瓶颈是否是两种文本表示的互补信息没有在单一路径中同时保留。

## 锁定公式

只使用 seed 66 validation 的八个 missing rates，在官方零阈值下搜索一个
全局凸组合权重，得到：

```text
p_ensemble = 0.71 * p_roberta_lora + 0.29 * p_deberta
```

该权重对所有 missing rates 和 seeds 67--70 固定。禁止：

- 使用 test label 调整权重；
- 按 rate 或 seed 单独调权重；
- 修改零分类阈值；
- 重训 Original 之外的额外模型或修改 GCNet 配置。

## 确认协议

- 数据集：CMU-MOSI；
- downstream seeds：66--70；
- 每个 seed 均使用 `train_rate_mode=all`、Slot fusion、100 epochs；
- DeBERTa 与 RoBERTa-LoRA 模型的 mask、labels 和 utterance 顺序必须逐元素一致；
- seed 66 两个模型直接继承；seeds 67--70 补齐配对模型；
- 每个 NPZ 独立重算 accuracy/W-F1。

通过门槛：五种子 miss=0 ensemble mean W-F1 至少 88.0，且至少 3/5 seeds
同时优于各自 DeBERTa 与 RoBERTa-LoRA 单模型。nonzero-rate mean 作为鲁棒性
副指标报告，不另行选参。

## 结论边界

该实验是 representation-complementarity diagnosis 和双模型 ensemble control，
不是新的单模型核心模块。若通过，它证明下一步应把两类文本证据压入一个可训练
的单模型融合层；在完成该压缩实验前，不能把 ensemble 结果写成主方法结果。
