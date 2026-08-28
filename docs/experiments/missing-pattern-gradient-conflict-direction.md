# Future Direction：Missing-Pattern Task Gradient Conflict

## 问题定义

同一个 mixed-rate batch 会同时产生多个 source-target tasks，例如：

\[
AV\rightarrow T,\qquad AT\rightarrow V,\qquad A\rightarrow T,\qquad A\rightarrow V.
\]

Target embedding、target-specific heads 与 MMoE routing 只提供隐式任务隔离。Observed modality projectors、GCNet backbone 与 shared experts 仍接收所有任务的联合梯度，因此可能出现负迁移。

## 研究问题

1. 六方向任务在 Projector、GCNet 和 shared experts 上的梯度余弦是否随 missing rate 改变？
2. 哪些 source-target pairs 长期冲突，哪些互补？
3. MMoE routing 是否真实形成任务专门化，还是所有任务集中到相同 experts？
4. 利用 source-target pattern 结构进行 conflict-aware routing/update，能否优于通用 PCGrad、CAGrad、GradNorm 与普通 MMoE？

## 最小证据链

```text
shared predictor
→ target heads
→ ordinary MMoE
→ pattern-conditioned routing
→ pattern-conflict-aware optimization
```

每个层级记录：六方向 loss、共享层梯度余弦矩阵、expert usage、各 pattern/rate 下游指标。

## 与当前工作的边界

当前 Single-View Missing-M3 论文只声称通过 source/target conditioning 与 MMoE 减少任务混叠，不声称消除梯度冲突。当前代码不得在主实验中临时加入 gradient surgery；本方向作为独立研究问题另行调研和验证。
