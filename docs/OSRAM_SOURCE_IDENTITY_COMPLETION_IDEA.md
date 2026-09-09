# OSRAM：保留观测/补全身份的三槽位融合

记录日期：2026-09-08

状态：用户要求先记录；尚未实现、尚未训练。不是已验证改进。

## 核心想法

MMoE 预测缺失模态 latent 后，让补全后的 A/T/V 三个槽位参与分类融合。
保留原始 availability mask，但在补全融合处将它解释为来源身份：
1 = 真实观测，0 = 预测补全，而不是继续用它删除补全槽位。

例如 mask=011（按 A/T/V 排列）：

| 槽位 | 内容 | 模态身份 | 来源身份 |
|---|---|---|---|
| A | MMoE 预测 Audio latent | Audio | 补全 |
| T | Student Text latent | Text | 观测 |
| V | Student Visual latent | Visual | 观测 |

## 拟议数据流

```text
实际不完整输入 + 原始 mask
    → Observed Student Projectors
    → 当前 masked mean → OSRAM（一次）→ h、Base/Gap

可见 Student latent + 对应 Base/Gap
    → MMoE → 缺失 latent 预测

各槽位选择：观测时用 Student latent，缺失时用预测 latent
    → 添加模态身份与观测/补全来源身份
    → 按 A/T/V 固定顺序拼接
    → 融合 MLP
    → 与 OSRAM h 结合
    → 情绪任务头
```

这里替换的是补全后的融合方式；第一遍 OSRAM 输入仍由真实可见信息构造。
本草案不要求再执行第二次 OSRAM。

## 训练与测试

- 训练、测试均保留 MMoE、三槽位构造与分类回灌路径。
- 训练期情绪损失可以沿回灌路径更新 MMoE；同时保留 JEPA 监督。
- 测试期不调用 EMA Teacher，也不计算 JEPA loss。
- 缺失位置的真实完整特征只可作为训练监督，不能用于构造分类槽位。
- 原始 mask 继续用于初始输入屏蔽、预测目标选择、OSRAM 的缺失条件；
  仅在补全后的融合处不再用它把预测槽位置零。

## 待确定与风险

- 来源身份仅说明观测或预测，不是置信度，不能保证模型正确处理错误补全。
- 模态身份、来源身份的编码尺寸，融合 MLP，以及与 h 的结合方式尚未锁定。
- 需核对 Student latent 与 MMoE 输出的空间、尺度、归一化是否适合共同融合。
- 完整 ATV 时三个槽位均为观测，不产生缺失预测任务。
- 旧主干上回灌的负面实验作为风险参考，不能代替这版 OSRAM 的实测。
- 下一步需用户继续推进本方案后再落实实现与实验；本次仅记档。
