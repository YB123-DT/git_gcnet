# Missing-M3 Raw-Residual SDR 设计规格

日期：2026-08-30  
状态：已完成并关闭（正式 gate 失败）

## 1. 研究问题

已经完成的整主干实验没有证明 SDR 优于 GCNet：

- `sdr-public` 的五种子验证八 rate 均值为 77.5849%，比配对 GCNet Control
  低 0.4565 个点；
- `sdr-public` 的测试八 rate 均值为 78.8696%，与 Control 的 78.8680%
  实质持平；
- 参数更多的 `sdr-paper` 验证和测试均更低。

但该实验让 SDR 只接收 Slot Observed-Set Encoder 压缩后的 256D node，而原始
SDR 路径接收 Audio、Text、Visual 拼接后的 2560D 特征。因此它只能否定
“256D Slot 后直接换 SDR”，尚不能判断 SDR 是否需要保留高维模态细节。

本实验只回答一个可证伪问题：

> SDR 在 Missing-M3 中没有增益，是否主要由进入 SDR 前的 2560D→256D 过早压缩造成？

## 2. 候选比较与正式选择

### A. Raw-only SDR

直接把官方缺失后的 2560D 特征送入 SDR。信息保留完整，但 Student Projector
不在分类梯度路径上，JEPA latent 与情绪识别容易脱节，因此不采用。

### B. Raw-Residual SDR-public（采用）

对每个 observed modality 计算 Student latent，并以零初始化 residual 写回原始
模态宽度：

\[
s_i^m=P_m(x_i^m),
\qquad
\widetilde x_i^m=a_i^m\left[x_i^m+A_m(s_i^m)\right].
\]

随后拼接：

\[
\widetilde x_i=
[\widetilde x_i^A;\widetilde x_i^T;\widetilde x_i^V]
\in\mathbb R^{512+1024+1024}=\mathbb R^{2560},
\]

并送入 `sdr-public`。它同时保留 raw modality information 与 Student 的分类梯度，
且不增加第二条推理分支。

### C. Raw + Slot 拼接

将 2560D raw 与额外 256D Slot 拼接。该方案同时改变输入信息和分支容量，无法把
收益归因于“避免过早压缩”，因此本轮不采用。

## 3. 精确架构

```text
official incomplete A/T/V feature [L,B,2560]
  -> modality-specific Student Projectors
       observed modality -> s_m [L,B,256]
       missing/padding    -> exact zero
  -> zero-init modality residual adapters
       A: 256 -> 512
       T: 256 -> 1024
       V: 256 -> 1024
  -> observed raw block + Student residual
       missing/padding blocks remain exact zero
  -> concatenate raw-residual blocks [L,B,2560]
  -> padding-safe public-formula SDR temporal backbone
       2-layer BiGRU(2560 -> 400)
       temporal RGCN -> HypergraphConv -> highConv
       post-graph BiGRU -> hidden [L,B,500]
  -> existing MOSI regression head

same Student latents + same SDR hidden + availability
  -> existing Contextual Missing-M3 Predictor
  -> existing EMA target and JEPA loss (training only)
```

`sdr-public` 是当前仓库实现的 padding-safe public-formula adaptation。它复现公开
公式语义，但由于上游公开代码在变长序列上直接处理 padded GRU，本方法不声称与
上游 padding bug 数值等价。

## 4. 初始化与泄漏不变量

Residual adapter 的最后一层 weight 与 bias 全零初始化。因此 optimizer 第一步前：

\[
\widetilde x_i^m=a_i^m x_i^m.
\]

必须同时满足：

1. 未观测模态和 padding block 始终精确为零；
2. 改变 missing block 的原始数值不得改变 SDR 输入、Student latent 或 logits；
3. Student Projector、adapter、SDR backbone 和 Missing-M3 predictor 均收到有限梯度；
4. `predict_missing=False` 时 predictor 与 EMA teacher 不执行；
5. 默认 `sdr_input_type=slot` 的参数、state-dict key、初始化 RNG 与现有结果兼容。

## 5. 唯一变化与保持项

唯一处理变量：

```text
Slot-SDR-public: 256D Slot -> SDR-public
Raw-SDR-public : 2560D Raw-Residual -> SDR-public
```

保持不变：CMU-MOSI split、冻结 wav2vec/DeBERTa/MANet 特征、五个 seeds、八个
missing rates、all-rates-per-batch、mask schedule、MSE 分类任务、JEPA weight、
EMA、MMoE predictor、optimizer、100 epochs、八 rate validation W-F1 选 checkpoint、
window 2/2、hidden 200、graph hidden 100、学习率 5e-4 和测试指标。

本轮不加入 reconstruction、attention、第二个 graph branch、completion、蒸馏、
ensemble、额外 loss 或参数搜索。`sdr-paper` 不再运行。

## 6. 代码边界

新版本放入独立目录 `gcnet_missing_m3_raw_sdr/`。它复用已经验证的：

- `gcnet_missing_m3.RawResidualObservedEncoder`；
- `gcnet_missing_m3_sdr_backbone.SDRConversationBackbone`；
- Missing-M3 训练、mask、评价与审计原语。

为避免复制 SDR 实现，只允许在共享 `MissingM3SDRModel` 增加一个默认值为 `slot`
的 `sdr_input_type` 配置；旧目录默认行为必须由回归测试锁定。新目录将该配置锁死为
`raw-residual` 和 `sdr-public`。

## 7. 实验矩阵

只训练五个新模型：

```text
seeds 66, 67, 68, 69, 70
每个 seed：一个 all-rates-per-batch checkpoint
每个 checkpoint：测试 miss 0.0, 0.1, ..., 0.7
GPU：2, 3, 5, 6, 7；禁止 GPU 4
```

因此是 5 个训练 job、40 个报告单元，而不是 40 次训练。现有 GCNet Control、
Slot-SDR-public 与旧 Raw-Residual GCNet 结果只继承，不重跑。

## 8. 判断规则

正式选择只使用 validation：

1. Raw-SDR 的五种子八 rate validation 均值高于 Slot-SDR-public；
2. 相对 Slot-SDR-public 至少 3/5 seed 为正；
3. high missing（0.4--0.7）validation 均值不下降；
4. 无非有限 loss、单符号输出或表示坍塌。

若同时超过配对 GCNet Control，则保留为候选；否则判定“恢复高维输入不足以使 SDR
适配 MOSI”。测试集只作 validation-selected checkpoint 的描述性报告，不用于选择。

旧 Raw-Residual GCNet 的测试八 rate 均值为 78.1762%，已经说明 raw residual 本身
不是普遍有效的模块。因此本实验只能支持或否定 `Raw-Residual × SDR` 的交互假设，
不得把任何提升归因于单独的 Raw-Residual 或单独的 SDR。

## 9. 交付物

- 独立 `gcnet_missing_m3_raw_sdr/` 模型、训练入口、runner、测试和状态文档；
- 单元测试、一次 1-epoch smoke 和五种子正式结果；
- config、history、metrics、mask/provenance SHA256 与 compact summary；
- Lore commit，并推送到 `YB123-DT/git_gcnet` 当前研究分支。
