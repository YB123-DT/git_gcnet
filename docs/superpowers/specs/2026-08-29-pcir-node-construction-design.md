# PCIR 图前联合节点构造设计

## 1. 目标与唯一变量

当前 Slot Missing-M3 将三个 observed Student latent 放入固定 A/T/V slot，拼接 pattern
embedding 后用一个 MLP 压缩为 GCNet node。本实验只在该 node 上增加一个
Pattern-Conditioned Interaction Residual（PCIR）；GCNet、Predictor、Teacher、Loss、mask
和训练协议全部不变。

## 2. 插入位置

```text
Frozen A/T/V features
→ Student Projectors
→ Original Slot Node
→ PCIR residual                  # 唯一新增部分
→ Original Temporal/Speaker GCNet
→ Emotion Head
```

PCIR 输出维度与 Slot node 相同，均为 `[L,B,256]`。它不预测缺失模态，不增加 graph
branch，也不改变 Predictor 的 Student latent 输入。

## 3. Pattern-conditioned 单模态校准

对七种有效 pattern `A,T,V,AT,AV,TV,ATV` 和三个 modality 保存逐维 scale/shift：

```python
scale_shift.shape == [8, 3, 2, 256]

if modality_is_observed:
    corrected = latent * (1 + scale[pattern, modality])
    corrected = corrected + shift[pattern, modality]
else:
    corrected = zeros(256)
```

全部 scale/shift 初始化为零。该步骤让 A-only Audio 与 AT/AV/ATV 中的 Audio 拥有不同
校准，但绝不读取 missing feature。

## 4. Observed-pair interaction

固定三对：`AT, AV, TV`。每个 active pair 构造：

```python
pair_input = concat(
    left,
    right,
    left * right,
    abs(left - right),
    pair_embedding,
)

pair = shared_pair_mlp(pair_input)
pair = pair * pair_is_observed
```

- A/T/V：没有 active pair，pair summary 为零；
- AT/AV/TV：只有对应的一对；
- ATV：AT、AV、TV 三对取均值。

`pair_embedding_dim=32`，shared pair bottleneck rank `64`。固定 slot 顺序已经区分左右模态，
不再计算反向重复 pair。

## 5. Residual 输出

```python
observed_summary = mean(corrected observed modalities)
pair_summary = mean(active pair outputs)  # 无 pair 时为零

residual = residual_mlp(
    concat(observed_summary, pair_summary, pattern_embedding)
)

final_node = original_slot_node + residual
```

PCIR 自有 `pattern_embedding_dim=32`，residual hidden dim `128`。`residual_mlp` 最后一层
weight/bias 全零初始化，因此初始化时：

```python
final_node == original_slot_node
```

训练后 ATV 也允许产生 residual，不强制完整模态永远等于 Original。

## 6. 开关与兼容性

新增：

```text
--node-interaction-residual
```

默认关闭。关闭时不实例化任何 PCIR 参数，旧 state-dict key、初始化 RNG 和输出保持不变。
开启时要求：

```text
fusion_type=slot
representation_type=slot
local_context_residual=false
classification_completion=false
```

PCIR 在所有 shared modules 初始化完成后实例化，保证相同 seed 下 Control 与 Treatment 的
共有参数逐 tensor 配对一致。

## 7. 参数预算

正式 `latent_dim=256`：

- pattern/modality scale-shift：12,288；
- pair embedding 与 low-rank shared pair MLP：约 85k；
- residual pattern embedding 与 544→128→256 MLP：约 102k。

预计新增约 0.20M 参数，远小于当前约 32M 总参数。正式结果记录实际参数差。

## 8. 测试要求

1. 七种 pattern shape 与 padding；
2. 初始化 forward 与 Original Slot 精确相同；
3. 修改 missing latent 不改变 residual；
4. A/T/V 没有 pair contribution，但 unary correction 有效；
5. AT/AV/TV 只激活一对，ATV 激活三对；
6. pair 对 source content 敏感；
7. backward 到 scale/shift、pair MLP、residual MLP、Student projector 和 GCNet；
8. 默认 key/RNG/output 回归；
9. Control/Treatment shared initialization 配对；
10. CLI/config/checkpoint provenance；
11. CPU/GPU FP32；
12. 参数数量。

## 9. 正式实验

- CMU-MOSI official split，fold1；
- seeds 66--70；
- `lr=5e-4`；
- Slot、Regression-MSE、all-rates-per-batch；
- 100 epochs；validation 八-rate W-F1 均值选 checkpoint；
- 一个 checkpoint 测八个固定 test masks；
- `5e-4` Slot Control 五种子直接继承，不重跑。

通过条件：

- test 八-rate mean delta > 0；
- high-missing `0.4--0.7` delta > 0；
- 两项均至少 3/5 seed 为正；
- miss0 mean 不下降超过 0.3；
- 40/40 mask SHA 配对，无坍塌。

## 10. 禁止变化

本轮不得新增 attention、可靠性 gate、缺失模态生成、test-time Predictor、额外 Loss、图层、
图边、scheduler 或上游 Encoder。不得根据 seed66 test 修改 PCIR 维度或公式。
