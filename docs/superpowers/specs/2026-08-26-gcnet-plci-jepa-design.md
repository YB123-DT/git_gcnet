# GCNet Source-Anchored Pattern-Lattice JEPA 设计

## 1. 目标与边界

PLCI-JEPA 以 GCNet 为对话图主干，用训练期跨模态 latent prediction 塑造其
缺失模态表示。方法不在测试期生成、补齐或回灌缺失模态。

本版本只保留四个核心机制：

1. modality-specific online student projector 与 EMA teacher projector；
2. source-only target-space anchor；
3. 当前 auxiliary observed-subset 的低秩 graph-context correction；
4. 双源条件下的低秩、有界 conditional innovation。

不加入 variance/covariance loss、contrastive loss、learned reliability gate、
path-consistency loss、MoE、额外图层或 test-time completion。

## 2. 训练视图

### 2.1 Natural view

Natural view 使用正式 rate-matched mask bank，保留原 GCNet 分类和
missing-only reconstruction：

\[
\mathcal L_{\mathrm{nat}}
=
\mathcal L_{\mathrm{cls}}^{\mathrm{nat}}
+
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}^{\mathrm{nat}}.
\]

只在不完整 utterance 上启用 student adapter 和 pattern hidden residual。
ATV utterance bypass 两者。若整段 conversation 均为 ATV，则固定共享参数下的
forward computation 与 Original GCNet 一致。

### 2.2 Pattern-balanced auxiliary view

对每个有效 utterance 独立采样：

\[
S_i^J\sim\operatorname{Uniform}\{A,T,V,AT,AV,TV\}.
\]

padding 不采样；natural mask RNG 与 auxiliary RNG 独立。同一 conversation
内允许不同 utterance 使用不同 pattern，整段 conversation 只进行一次
auxiliary GCNet forward。

Auxiliary view 从完整训练 feature 副本继续删除 modality。完整 target 只进入
no-gradient EMA teacher，不进入 student source、GCNet input 或 predictor input。

## 3. Natural-path adaptation

对模态 \(m\)：

\[
s_i^m=P_m^s(\operatorname{LN}_m(x_i^m)),
\qquad
A_m:\mathbb R^d\rightarrow\mathbb R^{d_m}.
\]

令 \(a_i^m\) 为 availability，且：

\[
\delta_i=1-\mathbb I[S_i=ATV].
\]

adapted feature 为：

\[
\widetilde x_i^m
=
a_i^m\left[x_i^m+\delta_i A_m(s_i^m)\right].
\]

Adapter 末层零初始化。缺失 block 始终为零；ATV utterance 不执行 projector
和 adapter。

Original pre-graph recurrent encoder 输出 \(r_{i,0}^S\) 后再加入 pattern：

\[
h_{i,0}^S
=
r_{i,0}^S+delta_iW_{\mathrm{pat}}e_{S_i}.
\]

\(W_{\mathrm{pat}}\) 零初始化，\(e_{ATV}=0\)。pattern signal 不进入 raw
Audio/Text/Visual blocks。

## 4. EMA target

Student/teacher projector 使用确定性结构：LayerNorm、Linear、GELU、Linear、
LayerNorm；不使用 BatchNorm 或 dropout。Teacher 初始精确复制 student，不进入
optimizer，并在每次 optimizer step 后更新：

\[
\theta_{P_q^t}^{k+1}
=
\tau_k\theta_{P_q^t}^{k}
+
(1-\tau_k)\theta_{P_q^s}^{k+1}.
\]

目标为：

\[
z_{i,q}^t
=
\operatorname{stopgrad}
\left[
\mathcal N_q(P_q^t(\operatorname{LN}_q(x_i^q)))
\right],
\]

其中 \(\mathcal N_q\) 为 no-affine LayerNorm 后的 L2 normalization。

## 5. Source-anchored predictor

### 5.1 Source-only anchor

Observed student latent 先 canonicalize：

\[
g_i^m=C_m(s_i^m).
\]

共享方向 trunk 与 target-specific output 产生：

\[
\beta_{i,m\to q}
=
O_q^\beta B_{\mathrm{shared}}([g_i^m;e_m;e_q]),
\qquad
b_{i,m\to q}=\mathcal N_q(\beta_{i,m\to q}).
\]

Base 不读取 GCNet hidden，因此同一个 \(A\to V\) anchor 可用于 A pattern 和
AT pattern 的 A-anchored path。

### 5.2 Current-subset graph context

一次 auxiliary forward 得到 \(h_i^J\)。低秩 raw correction：

\[
c_{i,S\to q}^{\mathrm{raw}}
=
U_q^c\sigma(V^ch_i^J+E_S^c+E_q^c),
\qquad r_c\ll d,
\]

其中 \(U_q^c\) 零初始化。

### 5.3 Conditional innovation

对 \(S=\{m,n\}\)，ordered relation：

\[
R_i^{n\mid m}
=
[g_i^m;g_i^n;g_i^n-g_i^m;g_i^m\odot g_i^n;e_{n\mid m};e_q;e_S].
\]

低秩 raw innovation：

\[
d_{i,n\mid m\to q}
=
U_q^\Delta\sigma(V_R^\Delta R_i^{n\mid m}+V_H^\Delta h_i^J),
\qquad r_\Delta\ll d,
\]

其中 \(U_q^\Delta\) 零初始化。

### 5.4 无参数 bounded residual

仅使用一个确定性结构约束，不增加 gate 或 penalty：

\[
\operatorname{BR}(v;\kappa)
=
\kappa\frac{v}{\|v\|_2+\epsilon}\tanh(\|v\|_2),
\]

所以 \(\|\operatorname{BR}(v;\kappa)\|_2\le\kappa\)。

## 6. 预测公式

单源：

\[
\widehat z_{i,q}^{\{m\}}
=
\mathcal N_q
\left[
b_{i,m\to q}
+
\operatorname{BR}(c_{i,\{m\}\to q}^{\mathrm{raw}};\kappa_c)
\right].
\]

双源的 m-anchored path：

\[
\widehat z_{i,q}^{m\rightsquigarrow n}
=
\mathcal N_q
\left[
b_{i,m\to q}
+
\operatorname{BR}(c_{i,S\to q}^{\mathrm{raw}};\kappa_c)
+
\operatorname{BR}(d_{i,n\mid m\to q};\kappa_\Delta)
\right].
\]

反向 path 使用 \(b_{i,n\to q}\) 与 \(d_{i,m\mid n\to q}\)，但共享当前
dual-source hidden 和 context correction。这里的 lattice 指完整 Boolean lattice
\(2^{\{A,T,V\}}\) 的 rank-1/rank-2 active subgraph；方法不声称执行 singleton、
dual 和 ATV 三套 backbone forward。

## 7. Loss

预测和 teacher 都是 unit-normalized，使用：

\[
\ell(a,b)=1-a^\top b.
\]

双源两条 path 分别对齐同一个 teacher，再求平均；不增加 path loss。单源 pattern
有两个 missing targets 时先在 utterance 内平均，避免获得双倍权重。

最终只有一个新增训练权重：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{cls}}^{\mathrm{nat}}
+
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}^{\mathrm{nat}}
+
\lambda_J\mathcal L_{\mathrm{JEPA}}^{\mathrm{aux}}
}.
\]

## 8. 六种 active pattern

| Pattern | Sources | Targets | Predictor behavior |
|---|---|---|---|
| A | A | T,V | 两个独立 A-anchored single predictions |
| T | T | A,V | 两个独立 T-anchored single predictions |
| V | V | A,T | 两个独立 V-anchored single predictions |
| AT | A,T | V | A-anchor 与 T-anchor 两条 bounded innovation paths |
| AV | A,V | T | A-anchor 与 V-anchor 两条 paths |
| TV | T,V | A | T-anchor 与 V-anchor 两条 paths |

表中的 pattern 描述当前 utterance；同一 auxiliary conversation 的其他 utterance
可使用不同 pattern。

## 9. 训练与测试保留项

训练顺序固定为 natural forward、auxiliary forward、一次合并 backward、
optimizer step、EMA update。Teacher requires_grad=False。

测试删除 teacher、base/context/innovation predictor 和 auxiliary sampler；保留
student projectors、classification adapters、pattern hidden projection、GCNet、
Original reconstruction head 和 classifier。不执行 test-time completion。

## 10. 泄漏不变量与诊断

若 \(q\notin S_i^J\)，改变完整 target \(x_i^q\) 不得改变 student latent、
auxiliary GCNet hidden、anchor、context、innovation 或 prediction；只能改变 EMA
teacher target 和 loss。

Std、effective rank、Real-vs-Shuffle、prediction cosine、context norm 和 innovation
norm只记录为诊断，不进入训练 objective。训练阶段拥有完整 A/T/V feature 是本
方法与 Original GCNet 共享的前提假设。

