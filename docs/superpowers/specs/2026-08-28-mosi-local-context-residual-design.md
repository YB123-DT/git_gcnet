# MOSI Local-Context Residual Fusion 设计规格

## 目标

在不修改 Missing-M3 predictor、EMA、JEPA loss、mask 或 mixed-rate protocol 的前提下，联合学习 utterance-local modality evidence 与 GCNet conversation context。该候选是 frozen-feature 路线的最后一个结构筛选；seed 66 的 `miss=0` W-F1 达不到 87.0 时不扩五种子。

## 已排除方案

- Raw-Residual：miss=0 五种子均值 85.34；
- GCNet time-attn：seed 66 miss=0 为 56.85；
- hidden=50/100：seed 66 为 85.70/85.46；
- prediction threshold oracle：五种子均值约 86.21；
- Slot 与完整 M3 prediction 的 test-oracle 线性融合：均值约 87.15。

因此新候选不能只是输出标量加权，而要在 representation level 联合训练。

## 结构

保留当前最佳 `fusion_type=slot`。Observed Student latents 为：

\[
s_i=[s_i^A;s_i^T;s_i^V;a_i]\in\mathbb R^{3d+3}.
\]

local residual：

\[
l_i=W_2\operatorname{GELU}(W_1\operatorname{LN}(s_i)),
\qquad l_i\in\mathbb R^{d_h}.
\]

其中 `W2` 的 weight 与 bias 全零初始化。GCNet 得到 conversation hidden：

\[
h_i^{G}=G_\theta(u_i).
\]

分类 hidden 与输出：

\[
h_i^{C}=h_i^{G}+l_i,
\qquad \hat y_i=C_\psi(h_i^{C}).
\]

M3 predictor 继续使用未融合的 `h_i^G`：

\[
\widehat z_{i,q}=P(s_i^m,h_i^G,m,q).
\]

这样唯一变化是分类表示的 local-context residual，不同时改变 missing-latent prediction task。

## 模块接口

```python
class LocalContextResidualFusion(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        ...

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:  # [L,B,context_dim]
        ...
```

Missing/padding Student latents 已为零；availability 三位向量显式加入输入。padding residual 必须精确为零。

`MissingM3GraphModel` 新增：

```python
local_context_residual: bool = False
local_fusion_hidden_dim: int = 256
local_fusion_dropout: float = 0.2
```

CLI：

```text
--local-context-residual
--local-fusion-hidden-dim 256
--local-fusion-dropout 0.2
```

第一轮只允许与 `fusion_type=slot` 联用。默认关闭，保证已有 mean/slot/raw-residual 模型行为不变。

## 不变量

- Local 模块在所有 shared modules 之后构造，不能改变已有参数初始化 RNG；
- 零初始化时，同一 shared state 下 logits、returned hidden 和 predictor output 与 Slot 精确一致；
- local residual 只进入 sentiment regression，不进入 M3 predictor；
- 不增加第二个 regression head、gate、attention、local auxiliary loss 或 reconstruction；
- 推理保留 Student、Slot encoder、GCNet、local residual 和唯一 regression head；Teacher/Predictor 不执行。

## 测试

1. local residual shape 正确，padding 为零；
2. missing feature value 修改不影响 local residual；
3. base/local 相同 seed 的 shared state 完全相同；
4. 零初始化 forward 与 Slot logits/hidden/predictions 精确一致；
5. local enabled backward 后 Student、local fusion、GCNet、predictor gradient 有限非零；
6. local disabled 不实例化新参数；
7. 非 Slot 联用时显式报错；
8. CLI/config/checkpoint 保存全部 local 配置；
9. 既有 22 tests 无回归。

## 实验

第一阶段：CMU-MOSI seed 66，Slot、hidden=200、time-attn=False、其余正式配置不变。

扩展门槛：

\[
W\mbox{-}F1_{seed66,miss0}\ge87.0.
\]

未通过：标记 frozen-feature architecture ceiling，停止继续叠加模块，下一阶段转 upstream encoder/LoRA。

通过：运行 seeds 66–70，正式门槛仍为 miss=0 五种子均值 ≥88，且 nonzero-rate mean 不低于 Slot 超过 0.5。

