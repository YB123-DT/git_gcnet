# 时序后接模块的工程接口：残差、维度、初始化与归一化

2026-10-22。只读代码与原论文审查；OSRAM 基础提交为 `a88bffbfc01a5de209d95869abed61eaba22456d`，分支 `feature/ecerc-features`。没有修改模型、训练协议或旧结果，没有启动训练、GPU 推理或 SSH 任务。用户进一步明确关注“残差和维度这种细节处理”，因此本文将工程接法放在主要位置。

## 1. 结论先行

差别不只是“别人有一个好模块、我们没有”，也不能统一归因于“残差压住了新模块”。

已核对的 DialogueGCN、MMGCN 和 M3NET 在关键模块出口保留**独立的拼接通道**；它们也使用内部残差。你的原 Nested 和 V3 residual 多数把新表示先映射回旧读取坐标，**加回证据后，再经过原 Flat 的融合和归一化**。这是不同的表示约束和梯度路径，不是同一条结构仅换了模块名字。

目前有证据的细节差异是：

1. 新旧表示在进入分类器之前是否已被相加，还是仍可分别加权。
2. 窄维度是每个 Head 的宽度，还是一整条读取角色汇总后的宽度。
3. 新模块与下游 Flat 的零初始化是否串联。
4. Norm 位于分支前、相加后，还是全拼接向量上；所谓“图归一化”不等于特征 LayerNorm。
5. 新分支最终经过哪个既有预测瓶颈，其尺寸是否真的扩大了独立输出自由度。

以上是可核对的结构差异，**尚不是长期无提升的单一因果解释**。历史上直接替换、concat、加宽和冻结主干都已有记录，不能把其中任何一项包装成从未尝试且一定有效的解法。

## 2. 别人的具体尺寸与连接

以下数字属于指定代码入口，不默认等同于论文全部配置。官方 HEAD 与本地补丁的差异保存在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

| 方法／入口 | 关键维度链 | add 与 concat 的位置 | 出口初始化／Norm |
|---|---|---|---|
| DialogueGCN，IEMOCAP，开启 graph-model | 文本100 → 双向 LSTM 200 → 图100；拼接200+100=300 → MLP100 → 6类 | 原时序状态与图输出 **concat**；图输出不需要先投回200再加原状态 | 没有显式零初始化图出口；活动路径无特征 LN/BN；分类器 Linear→ReLU→Dropout→Linear |
| MMGCN，官方 run.sh | 每模态200 → 图100；每模态拼接200+100=300；三模态后900 → 6/7类 | 图内混入初始状态；末端原节点与图结果 **concat** | 没有零初始化出口；邻接度归一化，不是特征 LN；传播层和最终分类前有 Dropout |
| M3NET，README 的 hyper/concat_DHT/avl | IEMOCAP 每节点512、MELD每节点1024 → 共同512；超图512与成对图512 → concat1024；三模态后3072 → 类别 | 成对图内部每层 **add**；两图分支及三模态出口 **concat** | 内部 add 后没有 LN；默认关闭额外原节点直通 concat；出口非零权重初始化 |
| MulT，三目标分支 | 每模态投影30；每目标两路 cross-attention 拼接60；三个目标末态拼接180 → 残差MLP180 → 预测 | Attention/FFN内部 add；跨方向与跨模态出口 concat；末端仍有 add | Transformer 子层是 Pre-LN；FFN 为 d→4d→d；Xavier 非零权重、零 bias，不是整个分支归零 |
| MFN | 3个 LSTM hidden 宽度之和为 H；前后 cell 拼接2H → attention/candidate → d_mem；最终 H+d_mem → MLP → 预测 | 三个模态末态与跨模态 Memory **concat**；Memory 内为两门控项相加 | 普通 Linear/LSTM 初始化，无零出口和显式 LN；h/c/m 初始状态为零，不等于参数零初始化 |

源码入口：

- [DialogueGCN 图与 concat](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueGCN/model.py#L789)，[脚本实例维度](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueGCN/train_IEMOCAP.py#L192)。裸 CLI 默认不开 graph-model，不能把表中数字当作所有运行的默认值。
- [MMGCN 图内混合](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/model_GCN.py#L163)，[末端 concat](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/model_GCN.py#L435)，[外部分类器](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/model.py#L1191)。`feature_fc` 虽声明但未用于该 return_feature 路径。另有源码／命令不一致：run.sh 传4层，MMGCN 构造分支却硬编码64层（model.py:1007）；不将此快照当成论文表5各深度的精确复现实例。
- [M3NET 两分支与出口](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/model_hyper.py#L153)，[最后分类](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/model.py#L865)。训练 CLI 的 use_residue=False 覆盖构造器默认值；这不关闭成对图内部残差。图度归一化不等于 LN。
- [MulT 拼接与残差](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L108)，[Pre-LN 与 FFN](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/modules/transformer.py#L125)。30维是这份公开实现的设定，不声称它是通用最佳维度。
- [MFN 维度与状态](https://github.com/pliang279/MFN/blob/b0453fb6e21b581c796246a0c32a6551f2028c1b/test_mosi.py#L183)，[递推及末态 concat](https://github.com/pliang279/MFN/blob/b0453fb6e21b581c796246a0c32a6551f2028c1b/test_mosi.py#L248)。H 和 d_mem 由配置确定，不凭代码末尾随机搜索空间猜某个预训练 checkpoint 的生效宽度。

## 3. 我们的具体接法

### 3.1 原 Flat

标准 cfg84：Local256；四路历史读取各1024，其中每路512是有效因果前向，另外512为零。实际接口是：

```text
[L256 ; B1024 ; GA1024 ; GT1024 ; GV1024] = 4352
  → LayerNorm(4352)
  → Linear(4352,1600) → GELU → Dropout
  → Linear(1600,1600)，该层 weight/bias 零初始化
  → 与 LocalSkip(256→1600) 相加
  → LayerNorm(1600)
  → 原任务头
```

Adapter 参数为 **9,535,104**；默认 V3 为 **335,006**。参数多不证明 Flat 最优，参数少不证明 V3 无力；但“加一个网络”是在既有大融合器之前进行受约束的修改，不是首次建立融合能力。[原定义](../../gcnet_missing_m3/osram.py#L1165)、[已有参数核验](../osram_module_connection_audit_20261022/INITIAL_GRADIENTS.json)。

原始数值包含2304个非固定零槽位，但**不能把原 LayerNorm(4352) 直接换成 LayerNorm(2304) 当作等价压缩**。固定零槽也参与均值和方差；经过仿射 LN 后不一定为零。更不能据此直接把所有对应权重宣布为“完全无用”。

### 3.2 原 Nested 的64与 V3 的64不是同一含义

| 项目 | 原 Nested | 默认 V3 |
|---|---|---|
| 最初的历史输入 | 每个角色512拆为8个真实64维 Head | 每个角色×来源各512维 |
| 内部64的含义 | 每个 Head token 为64维，8个分别保留；另有 Local256→64 | 每条来源读取512→64；最后每个角色三个来源汇总为一个64维向量 |
| 返回历史接口 | 每个 Head 独立64→64 Decoder，再拼回512 | 每个角色一个64→512 Decoder |
| 外部接法 | Local/历史输入残差；Local Skip 仍用原 Local | 只校正 Base/Gap；原 Local 和 Local Skip 不改 |
| 内部 Norm | tokenizer LN；GIN 层后 LN | PairMLP 输入 LN；多来源 self-attention add 后 LN；最终 cross-attention 后无独立 LN |

证据：[HeadTokenizer](../../gcnet_missing_m3/meaningful_blocks_common.py#L12)、[TokenAdapter](../../gcnet_missing_m3/meaningful_input_new40.py#L18)、[Nested core](../../gcnet_missing_m3/meaningful_new40_structure.py#L323)、[V3](../../gcnet_missing_m3/source_role_v3.py#L28)。

因此，“Nested只有64维”不能解释成“全部512维历史被统一压成64”；而 V3 每个角色最终确实经过一个64维出口。

固定训练后的一个角色 Decoder，激活样本的校正量为 `delta_i=W z_i+b`，W 为512×64。任意两个激活样本的差为 `W(z_i-z_j)`，因此这些差异的秩最多64。映射回512维不会恢复512个独立变化方向。带 bias 的未中心化输出，以及混入不激活时的全零输出，安全上界是65；不能把64上界套在加回原证据、非线性 Flat 输出或不同 checkpoint 的合集上。

这是结构约束，不证明64一定太小。更宽的历史实验也没有自动给出稳定优势；MulT 的窄通道更表明必须结合 token 数、出口数和后续连接理解宽度。

## 4. add 与 concat 到底差在哪里

把二者只称“残差”会掩盖关键差异：

```text
独立拼接： old x ───────────┐
                         ├→ [x ; z] → 分类器分别学习 W_x、W_z
            new z ───────┘

输入相加： old C ──────────┐
                         ├→ C + Decoder(z) → 原 Flat → 预测
            new z → D ───┘
```

线性读出的小例子中，`W[x;z]=W_x x+W_z z`，而 `W(x+z)=Wx+Wz`。后者在融合后共享同一坐标解释，且下一层仅看到和，不能一般性地恢复两路各自身份。实际 OSRAM 有 LN/GELU，并非这个线性例子的完整等价，但“先相加再交给分类器”这一信息接口差异成立。

V3 的 Decoder 必须学到**与原 Base/Gap 坐标相容的校正**，不是仅输出一套能被新分类器单独使用的特征。学习投影能够实现这种对齐，并非错误设计；它减少接口扩张、保留初始基线，但限制了自由度。

同样不能反推 concat 必胜：拼接增加参数和可用通道，也增加过拟合与优化负担。仓库已有 [Nested concat 记录](../osram_nested_concat_20261011/STATUS.md)，包括容量对照；其 [方向复核](../osram_nested_concat_20261011/DIRECTION_REVIEW.md) 记录的是当时的中途状态，不拿它冒充最终结果，更不重新把 concat 宣称为新点子。

## 5. 两道零初始化：真实存在，但不等于长期学不动

设原 Adapter 简写为 A(x)=W2·phi(W1 x)，新增输入校正为 D·z。V3 residual 从头训练时 W2=0、D=0：

- 首次任务反传：W2 可以有梯度，但通过 A 传给 V3 Decoder 的梯度为零。
- W2 更新后：D 可以得到梯度，而 V3 内层仍受 D=0 阻挡。
- D 更新后：PairMLP、Attention、Query 可以获得任务梯度。

这里说的是任务梯度；不能据此声称带 weight decay 的优化器绝不改变任何其他参数。预训练非零 Flat、隐藏层后残差、随机 Decoder 等情况也不具有完全相同的启动顺序。

此前用实际类、合成输入、CPU、dropout=0、weight_decay=0 的四步检查已记录：

| backward | Flat 最后层梯度范数 | V3 Base Decoder | V3 PairMLP |
|---:|---:|---:|---:|
| 1 | 7.639775 | 0 | 0 |
| 2 | 7.064045 | 0.284228 | 0 |
| 3 | 5.017679 | 0.313444 | 0.101972 |
| 4 | 8.913265 | 0.634138 | 0.220265 |

[原始记录](../osram_module_connection_audit_20261022/INITIAL_GRADIENTS.json)。本轮只复查记录，没有再运行该实验。这不是50/150轮真实训练轨迹，也没有证明初始化延迟是性能瓶颈。DialogueGCN/M3NET/MFN 没有这类显式关闭出口；MulT 的 bias 为零但权重非零，二者不可混同。

## 6. Norm、尺度和梯度长度

### Norm 不是一个可以互换的位置标签

- MulT 子层是 `x + F(LN(x))`，局部残差包含不经过该子层 LN 的恒等路径；编码器仍有末端归一化，不能说整网梯度始终无 Norm。
- V3 的多来源 self-attention 是 `LN(x+Attention(x))`；只有一个有效来源时这一步直接旁路。此条件化 Norm 是实际设计，不应描述成所有来源数都同一路径。
- V3 的512维校正先加到历史读取，再统一过 Flat 输入 LN；Flat 最后还有 `LN(S(Local)+Adapter(...))`。
- DialogueGCN/MMGCN 这里没有同样的特征 LN；图的邻接归一化不能替代这项比较。

LayerNorm 会严格消去沿全1向量的共同平移；忽略 epsilon 时，对正的整体尺度变化不敏感。这能解释“范数明显变了但任务输出变化不大”在数学上可能发生，**但不是说所有 Memory 缩放都被抵消**：只放大 B/G 而保留 Local 会改变拼接向量内部比例，并非整个 LN 输入的统一缩放；其他方向的修正仍可能改变归一化结果，但能否影响预测取决于下游映射。

### 范数比例不是决策贡献比例

对于固定权重的小校正，局部近似为：

`delta_y ≈ J_head · J_emotion_norm · J_adapter · delta_C`。

因此 residual/input 范数1%或10%并不能直接换算成 ACC 或 W-F1 影响。需要经过真实下游 Jacobian；近分类边界小改变也可能翻转类别。这里不是新发现模型“必须大残差”，也不是要求再跑范数诊断。

两种旁路也须分清：参考模型的原表示 concat 仍让新特征有独立输入列；你的 Local Skip 绕过了 Adapter。前者也可能被忽略，后者也不必然排斥历史；不能靠有无旁路断言训练一定偏向某边。

## 7. 不要把“别人有效”当成普遍事实

以下只引用论文内部对照，**不与 OSRAM 八率成绩直接横比，也不从这些消融归因到某一种 Norm/add/concat**：

| 论文内部比较 | 原结果 | 含义／边界 |
|---|---|---|
| DialogueGCN Table5，IEMOCAP F1 | 顺序编码55.30；顺序+speaker图64.18 | 图在其协议中有明显价值；不能认定全由concat造成 |
| COGMEN Table6，IEMOCAP4，Text-only | 无GNN81.18；完整81.55，+0.37pp | 同一个图模块在该输入条件下增益很小 |
| COGMEN 同表，ATV | 无GNN80.28；完整84.50，+4.22pp | 增益随可用模态条件变化，不是只看是否“接了GNN” |
| MMGCN Table4，IEMOCAP | late fusion64.62；gated64.45；MFN62.77；MulT62.37 | 更复杂融合不自动比简单融合好 |
| MMGCN 同表，MELD | late58.26；gated58.18；MFN58.21；MulT57.93 | 其论文也存在加复杂机制无收益的结果 |

来源：[DialogueGCN 原文](https://aclanthology.org/D19-1015.pdf)、[COGMEN 原文](https://aclanthology.org/2022.naacl-main.306.pdf)、[MMGCN 原文](https://aclanthology.org/2021.acl-long.440.pdf)。MMGCN 不同融合方法并非已认证的等参数、等算力实验；不能将表中结果当作纯算子因果消融。COGMEN 本轮只核对论文，未核对本地实现。

此外，DialogueGCN/M3NET 保留逐句话节点，而 V3 的 `reshape(T*B*4,3,width)` 将时间变成批维，Attention 只在当前读取的三个来源之间运行。MMGCN 默认 A/V 只有 Linear，图同时承担它们的首次时序聚合。MFN 则在每个话语内的词级时间步更新跨模态 Memory。它们不是在同一种“已经完成联合记忆读取”的接口上添模块；但这些访问差异也不能证明 OSRAM 的写入一定有问题。多数对话图参考还允许未来上下文，不能将其收益承诺给因果 CHM。

## 8. 对“为什么我做那么多仍不涨”的准确判断

**不是所有失败都能归为残差，但多个不同模块确实被相同的窄出口、坐标相加、既有 Flat 和优化路径共同约束。** 因而换了内部图或 Attention，不一定改变最终能贡献的表示形式。这是值得纳入解释的共同因素，而不是已查明的唯一原因。

同时保留反证：

- 同 checkpoint 中，Source-Role 学到的 gamma 比强制 gamma=1 有任务增益，但新模型仍未超过独立 Flat；所以“最终不涨”等于“新增分支完全没作用”是错误的。[已有 MOSEI 配对分析](../osram_mosei_source_role_switch_20261021/completed/INTERPRETATION.md)。
- 原 Nested 没有把整路512压成64；直接替换、冻结原路径和独立历史访问也有失败记录。[上一轮代码与结果审查](../osram_module_connection_audit_20261022/RESULT.md)。
- 零初始化不是永久断梯度；去掉初始化保护、改为 concat、去掉 LN 或统一加宽都不能据此保证提升。

若以后比较接法，应该保持**同一个核心模块与输入**，只改变一个已明确的接口约束，并复用历史对照；本次没有新建这样的实验。当前交付是工程差异定位，而不是继续扩充候选池或宣布新方法有效。

## 9. 核验范围

- 官方源码按固定 Git HEAD 比较；MMGCN/M3NET 本地有因果补丁，不用于代表论文默认实现。
- 参数量和四步梯度数字来自已有可追溯记录，没有把合成输入称为真实训练结果。
- 文献元数据见 [paper_bank.json](paper_bank.json)，论文内部数字见 [paper_comparisons.csv](paper_comparisons.csv)。
- 仅新增本目录的双语分析与来源记录；未修改模型、checkpoint、数据、Mask、运行任务，也未复现参考论文性能。
