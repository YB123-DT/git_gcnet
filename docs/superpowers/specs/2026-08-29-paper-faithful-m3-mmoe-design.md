# Paper-Faithful M3 MMoE 设计规格

## 1. 目标与边界

本实验只替换 Single-View Missing-M3 中的 MMoE predictor primitive。Observed-Set
Encoder、GCNet、EMA Teacher、SmoothL1/InfoNCE、all-rates-per-batch、数据划分和
checkpoint selection 全部保持不变。现有 `dual-gate` 结果直接作为控制，不重新训练。

论文 M3-JEPA 明确为 regularization 与 contrastive 两个任务设置两个 gate；公开代码还
包含 task embedding、分支 LayerNorm 和 residual。当前实现只有双 gate 和共享 experts，
缺少后三项。本轮组合论文与公开代码中一致且可迁移的部分，不采用公开代码中与论文
冲突的单 gate。

## 2. Predictor 结构

对方向条件表示：

\[
c=\operatorname{LN}(s+W_hh+e_{src}+e_{tgt}).
\]

两个分支分别构造：

\[
x_{reg}=c+e_{reg},\qquad x_{cl}=c+e_{cl}.
\]

共享 expert 参数，但每个分支以自身输入独立计算 expert output，并由独立 gate 路由：

\[
m_b=\sum_{n\in\operatorname{TopK}(G_b(x_b))}G_b(x_b)_nE_n(x_b),
\quad b\in\{reg,cl\}.
\]

遵循公开代码，Top-K 权重来自完整 softmax 后的选中概率，不在 Top-K 内二次归一化。
分支输出为：

\[
u_b=x_b+\operatorname{GELU}(\operatorname{LN}_b(m_b)).
\]

最后沿用当前三模态 target-specific heads。source/target embedding 是从官方双模态方向
扩展到 A/T/V 六方向所必需的适配。

## 3. 兼容性

新增 `mmoe_variant`：

- `dual-gate`：当前实现，默认值；参数 key、初始化顺序和输出语义不变；
- `paper-faithful`：双 gate + task embedding + branch norm/residual。

旧 checkpoint 继续用 `dual-gate` 严格加载。正式 treatment 显式使用
`--mmoe-variant paper-faithful`。

## 4. 路由诊断

两个变体都累计非持久 buffer：每个分支/专家的 Top-K selection count、probability mass
和 token count。由此计算 usage distribution、mean routing mass 和 entropy。统计只记录，
不加入 load-balancing loss，避免一次实验同时改变结构与目标函数。

## 5. 验证与实验

单元测试锁定：默认 key 兼容、两个独立 gate、共享 experts、task embedding、分支 norm、
residual、官方 Top-K 权重、有限 backward 和路由统计。之后在远程官方环境执行 focused
tests 与一个真实 MOSI batch。

正式实验为 CMU-MOSI、Slot、Regression-MSE、all-rates-per-batch、seeds 66--70。每个
seed 训练一个 checkpoint 并测试八个 rate；当前五种子 `dual-gate` 结果继承。主判断为
八-rate 均值以及 0.4--0.7 均值；同时要求 miss0 不明显下降、至少 3/5 seeds 的高缺失
均值为正，并检查 expert starvation。

