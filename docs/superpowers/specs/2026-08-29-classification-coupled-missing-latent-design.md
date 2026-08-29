# Classification-Coupled Missing-Latent 设计规格

## 1. 研究问题

Training-only Missing-M3 的 Predictor 即使恢复 paper-faithful 分支结构，也没有提高 MOSI。
本实验检验更直接的因果链：让同一个 Missing-Latent Predictor 在训练与推理时均参与
emotion representation，而不是只提供训练期辅助梯度。

## 2. 单次 forward

```text
official incomplete input
→ Observed-Set Encoder
→ GCNet hidden h
→ Missing-Latent Predictor z_hat_q
→ target-specific residual projection
→ h_final = h + residual
→ emotion head
```

Predictor 仍只读取 observed Student latent、GCNet hidden、source/target identity 和显式
availability。真实缺失 target 只在训练期进入无梯度 EMA Teacher。

对实际缺失目标集合 `Q_i`：

\[
r_i=\frac{1}{|Q_i|}\sum_{q\in Q_i}
\tanh\left(W_q\operatorname{LN}_q(\widehat z^{reg}_{i,q})\right),
\qquad h_i^{final}=h_i+r_i.
\]

三个 `W_q` 零初始化。训练开始时 treatment 与 control 分类 hidden 相同；后续 `W_q`
和 Predictor 同时接受 emotion gradient。双缺失目标先平均，避免 A/T/V singleton 因目标
数量为二而放大 residual。

## 3. 推理合同

推理保留：Observed Student、GCNet、Missing-Latent Predictor、residual projection、Emotion
Head。推理删除：EMA Teacher、完整 target feature、SmoothL1/InfoNCE 和 EMA update。

ATV 没有缺失目标，因此 residual 精确为零；padding residual 也为零。Predictor 输出不作为
评估 artifact 返回，但其 forward 必须执行于不完整输入。

## 4. 唯一变量与实验

新增 `classification_completion: bool = False`，默认旧路径与旧 checkpoint 不变。
Treatment 显式开启，MMoE 固定为当前 `dual-gate`，避免与刚完成的 MMoE fidelity A/B
混合变量。

正式协议继承 CMU-MOSI Slot、Regression-MSE、all-rates-per-batch、100 epochs、seeds
66--70。Control 五种子直接继承；每个 treatment checkpoint 测试八个 rate。

通过标准：八-rate 和 0.4--0.7 均值均为正，至少 3/5 high-missing seed 为正，miss0 不
下降超过 0.3。未通过则说明当前 M3 latent 不适合直接作为分类补充证据。

