# MOSI Raw-Residual Missing-M3 设计规格

## 目标

当前 Missing-M3 在进入 GCNet 前将完整的 Audio/Text/Visual 特征从：

\[
d_A+d_T+d_V=512+1024+1024=2560
\]

压缩为一个 256D observed-set node。Slot Fusion 改善了不完整条件，却在 CMU-MOSI `miss=0` 上仅达到 85.50 W-F1，距离五种子均值 88 的目标约 2.50。

本轮只替换 Online Incomplete Encoder 的 node construction：保留原始 observed feature blocks，并让 Student latent 通过零初始化 residual adapter 参与分类。M3 predictor、EMA teacher、GCNet、loss 和 mixed-rate protocol 全部保持不变。

## 候选方案与决定

### A. Raw-only GCNet + 独立 Student（不采用）

直接把官方不完整原始特征送入 GCNet，Student 仅服务 JEPA predictor。该路径信息保留最完整，但 Student 与分类路径只通过 predictor context 间接耦合，不能保证 learned modality latent 改善分类表示。

### B. Raw-Residual Missing-M3（采用）

对每个 observed modality 保留原始特征，并加入 Student latent 的 modality-specific residual：

\[
s_i^m=P_m(\operatorname{LN}_m(x_i^m)),
\]

\[
\widetilde x_i^m=a_i^m\left[x_i^m+A_m(s_i^m)\right].
\]

拼接后送入原始输入宽度的 GCNet：

\[
\widetilde X_i=[\widetilde x_i^A;\widetilde x_i^T;\widetilde x_i^V]
\in\mathbb R^{2560}.
\]

该方案同时满足：原始信息不丢失；Student 受到分类梯度；missing blocks 仍为零；GCNet 主干接口恢复为原始 feature width。

### C. Raw + fused latent 拼接（不采用）

将 2560D raw feature 与额外 256D fused latent 拼接。它保留信息，但重新引入一个独立 fused node，扩大 recurrent input 并混淆收益来自 raw preservation 还是额外 latent branch。

## RawResidualObservedEncoder

新增：

```python
class RawResidualObservedEncoder(nn.Module):
    def __init__(
        self,
        dimensions: tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        ...

    def forward(
        self,
        features: torch.Tensor,      # [L,B,Da+Dt+Dv]
        availability: torch.Tensor,  # [L,B,3]
        umask: torch.Tensor,         # [B,L]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...
```

三个 Student Projector 与现有实现相同。每个 residual adapter 为：

```text
LayerNorm(latent_dim)
→ Linear(latent_dim, modality_dim)
```

adapter 的 Linear weight 与 bias 全零初始化。不加入 gate、attention、额外 MLP 或 residual cap。

forward 对每个 modality 执行：

```python
selected = valid & availability[..., modality_index].bool()
latent[selected] = projector(block[selected])
adapted_block[selected] = (
    block[selected] + adapter(latent[selected])
)
```

未 observed 或 padding 的位置保持精确零。不得依据 feature 数值是否为零推断 availability。

## MissingM3GraphModel 路由

扩展现有参数：

```text
fusion_type = mean | slot | raw-residual
```

- `mean`、`slot`：继续使用 `ObservedSetEncoder`，recurrent input 为 `latent_dim`；
- `raw-residual`：使用 `RawResidualObservedEncoder`，保留 `GraphModel` 在 `super().__init__()` 中建立的 `adim+tdim+vdim` recurrent input，不覆盖 `self.lstm/self.gru`。

两个 encoder 都暴露：

```python
encoder.projectors
```

因此 EMA teacher 与现有 `encode_teacher_targets()` 不变。

## 训练与推理

Natural mixed-rate view：

```text
official incomplete raw feature
→ modality Student Projector
→ zero-init raw residual adapter
→ original-width GCNet
→ MOSI regression
```

训练期 missing-latent path：

```text
observed Student latents + same GCNet hidden
→ existing M3 six-direction predictor
→ missing target EMA latent
→ existing JEPA loss
```

测试时：

- 保留 Student Projector、residual adapter、GCNet 和 regression head；
- 删除 Predictor 与 EMA Teacher 的 forward；
- 不生成或回灌缺失模态。

## 初始化不变量

在 adapter 初始化完成后、任何 optimizer step 之前：

\[
A_m(s_i^m)=0,
\]

因此：

\[
\widetilde x_i^m=a_i^m x_i^m.
\]

这保证训练起点的 GCNet classification input 与官方不完整 raw input 精确一致。训练后 adapter 可以改变 observed blocks，因此不声称最终模型等于 Original GCNet。

## 保持不变

- Dataset split、feature extractor 与 feature dimensions；
- mixed-rate batch schedule 0.0–0.7；
- 同一 checkpoint 测试八个 rates；
- mask generation、seeds 与 paired SHA256；
- Student Projector 结构；
- M3 Top-2/4-expert six-direction predictor；
- EMA tau、JEPA weight、temperature；
- Temporal/Speaker GCNet 与 regression head；
- optimizer、epochs、checkpoint selection；
- 无 reconstruction、无第二 view、无 test-time completion。

本轮不增加 pattern embedding、modality attention、gradient-conflict optimizer、rate embedding、parameter-matched control 或新的 loss。

## 测试

1. 七种 pattern 下 raw-residual output shape 为 `[L,B,2560]`，Student latent 为 `[L,B,256]`。
2. 零初始化时 output 与 `features * expanded_availability` 精确一致。
3. 修改 missing feature values 不得改变 output 或 observed Student latents。
4. padding output 与 latents 全零。
5. backward 后梯度到达 Student Projector、adapter、GCNet 和 predictor。
6. `raw-residual` 模型 recurrent `input_size == adim+tdim+vdim`；mean/slot 仍为 `latent_dim`。
7. mean/slot 原测试与 checkpoint key 行为无回归。
8. CLI/config/checkpoint 明确记录 `fusion_type=raw-residual`。
9. CPU 单测与远程 GPU MOSI forward/backward/EMA 均为有限值。

## 正式实验

- Dataset：CMU-MOSI official split；
- Seeds：66、67、68、69、70；
- 每 seed 一个 mixed-rate checkpoint；
- 同一 checkpoint 测试 missing rates 0.0–0.7；
- GPU：0、1、2、3、5，禁用 GPU 4；
- mean 与 slot 结果直接继承，不重新训练；
- 保存 config、100-epoch history、metrics、40 个 prediction NPZ 和五个远程 checkpoint SHA256。

## 判断门槛

Primary：

\[
\operatorname{MeanSeedW\mbox{-}F1}(miss=0)\ge 88.0.
\]

Secondary：

- 七个 nonzero rates 的平均 W-F1 不得比 Slot 低超过 0.5；
- 无 NaN、无单一输出坍塌；
- 40/40 test masks 与 Slot/Mean 同 seed-rate 严格匹配；
- 至少 3/5 seeds 的 miss=0 W-F1 高于 Slot。

若 Primary 未通过，则 Raw-Residual 只能作为机制诊断，不得称为最终 MOSI 方案。下一轮必须根据 raw path、adapter norm、validation trajectory 的证据定位，不直接叠加新模块。

