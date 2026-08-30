# MOSI 等参数 SDT-style 对话主干诊断设计

## 1. 研究目标

当前局部 downstream 改造已在冻结 wav2vec、DeBERTa、MANet 特征的 MOSI
协议上进入平台期。本实验只回答一个问题：

> 在输入、缺失协议、Observed-Set 编码、Missing-M3 预测和训练目标完全相同的
> 条件下，整体移除 GCNet 对话主干并换成全上下文 Transformer，能否稳定提高
> 八个 missing rates 的表现？

本实验是结构归因诊断，不把普通 Transformer 声称为方法创新。SDT 与 CSS 已使用
全上下文注意力、位置编码和 speaker embedding；本实验明确记录这一级机制重合。

用户已要求持续自主执行，并要求新建独立版本目录、完成后上传 GitHub。本规格经过
自检后直接进入实现，不等待逐阶段确认。

## 2. 候选比较

比较过 3 种替换方式：

1. **单路等参数全上下文 Transformer（采用）：** 只接收现有 Slot observed-set
   node。它仅替换对话主干，能够直接检验 GCNet 是否为瓶颈。
2. **完整 SDT 九路 intra/inter-modal Transformer（拒绝）：** 需要重新展开
   T/A/V 三条模态流，同时改变 node construction 与对话主干，无法归因。
3. **Temporal CNN + Transformer（拒绝）：** SDT 代码中的 `Conv1D` 使用
   `kernel_size=1`，本质是通道投影。加入真正的时序 CNN 会引入第二个机制变量。

因此，本轮使用「SDT-style」只表示以下可追踪来源：全上下文注意力、正弦位置编码
和 speaker embedding；不表示复现完整 SDT。

## 3. 独立目录边界

新增目录：

```text
gcnet_missing_m3_sdt_backbone/
├── __init__.py
├── model.py
├── train_gcnet.py
├── run_mosi.py
├── README.md
├── STATUS.md
├── tests/
└── results/
```

- 现有 `gcnet_missing_m3/` 继续作为锁定 control，不改变其默认行为。
- 新目录复用现有数据加载、mask schedule、Observed-Set、Missing-M3 loss 与评估函数。
- 新目录拥有独立模型入口、训练入口、runner、测试和结果清单。
- GitHub 只提交源码、配置、日志摘要、JSON/Markdown 结果与必要的小型测试资产；不提交
  checkpoint、大型 NPZ 或数据集。
- 训练未完成时，`STATUS.md` 明确标记 `IN PROGRESS`；正式归档后改为 `COMPLETE`
  或 `CLOSED — NO IMPROVEMENT`，不能混入已完成版本。

## 4. 架构

### 4.1 修改前

```text
Slot observed-set node [L,B,256]
→ 2-layer BiLSTM
→ Temporal RGCN + GraphConv + branch BiLSTM
→ Speaker RGCN + GraphConv + branch BiLSTM
→ branch addition
→ hidden [L,B,250]
```

### 4.2 修改后

```text
Slot observed-set node [L,B,256]
→ Linear(256,384)
→ sinusoidal position encoding (max_len=512)
→ speaker embedding
→ 5 × Pre-LN Transformer encoder
   ├── 8-head full-context self-attention
   ├── FFN 384→704→384
   └── dropout
→ final LayerNorm
→ Linear(384,250)
→ ReLU
→ hidden [L,B,250]
```

Attention mask 只屏蔽 padding，不使用 causal triangular mask。因此每个有效 utterance
可以读取同一 conversation 内的过去和未来，这与当前 GCNet 的双向 BiLSTM 语义一致。

speaker index 从显式 `qmask [B,L]` 得到；有效位置存储 `0..n_speakers-1` 的 speaker
ID。padding 使用独立的 padding index，但 padding
输出最终严格置零。模型不得从特征数值是否为零推断 speaker、padding 或缺失状态。

### 4.3 输出接口

新模型继续输出：

```text
logits, classification_hidden, observed_latents, missing_predictions
```

其中 `classification_hidden` 形状仍为 `[L,B,250]`。因此以下组件原样复用：

- Shared emotion/regression head；
- Contextual Missing-M3 Predictor；
- EMA Teacher；
- classification loss 与 JEPA loss；
- checkpoint 选择和八 rate 评估。

## 5. 参数预算

锁定 MOSI control 使用 `hidden=100`、`graph_hidden=50`、Slot node 256D、
`time_attn=False`。代码审计将参数分为：

- **active-forward backbone：** 实际进入 LSTM、图卷积和图后 BiLSTM/Linear 前向的
  参数；
- **registered-inactive parameters：** 官方类中已注册但该配置不执行的 GRU 与
  `time_attn=False` 下的 MatchingAttention 参数。

Control 的 active-forward backbone 为 `5,864,700` 参数。候选配置的 registered
参数为：

```text
input projection                         98,688
speaker embedding (MOSI, 2 rows)            768
5 Transformer layers                 5,673,280
final LayerNorm                            768
output projection                       96,250
------------------------------------------------
total                                 5,869,754
```

其中 padding speaker row 的 384 个参数按设计始终零梯度，因此候选 effective
active-forward 参数为 `5,869,370`。它与 Control 的差值为 `+4,670`，即
`+0.080%`。正式实现后必须由官方 PyTorch 环境同时核验 registered 与 effective
口径。

新模型不保留无效 GRU、无效 MatchingAttention 或 dummy parameters。总注册参数会低于
Control，但实际参与前向的容量严格近似匹配。报告必须同时给出 registered、trainable、
active-forward 3 种口径，不能只选有利口径。

构造时先按 Control 的原始顺序初始化共享模块，再删除旧 LSTM/GRU/graph objects，最后
实例化新 backbone。测试必须证明相同 seed 下 Observed-Set、Teacher、Missing-M3
Predictor 和分类头的初始 tensor 与 Control 完全一致，避免 RNG 消耗差异污染比较。

## 6. 保持不变

以下内容全部锁定：

- wav2vec、DeBERTa、MANet 冻结特征及 feature dimensions；
- MOSI split、训练 seed 66–70 和数据顺序；
- Slot Observed-Set Encoder 及显式 availability；
- EMA Teacher 和 Missing-M3 Predictor；
- dual-gate MMoE、target-balanced JEPA、MSE task loss；
- `all-rates-per-batch` 训练与 8 个 missing rates；
- natural mask schedule、mask seed 和测试 mask；
- LR `5e-4`、weight decay `1e-5`、100 epochs、gradient clipping；
- 以 8-rate validation mean W-F1 选择最早最佳 epoch；
- 测试指标和 non-collapse 检查。

不组合新的 loss、completion、readout、CNN、LoRA、蒸馏或图模块。

## 7. 错误处理与不变量

- `input` 必须为 `[L,B,256]`，`umask` 与 speaker-ID `qmask` 必须为 `[B,L]`。
- `umask` 必须为二值连续前缀，且与 `seq_lengths` 一致。
- 每个有效 utterance 的 speaker ID 必须是有限整数且位于
  `[0,n_speakers-1]`；padding ID 被忽略并映射为 padding index。
- sequence length 超过 positional buffer 时立即报错，不静默截断。
- 所有有效输出、loss 和 gradient 必须有限。
- 改变 padding feature 数值不得改变有效位置输出。
- 新 runner 只接受唯一 treatment，拒绝同时启用旧候选开关。
- Existing Original 结果直接继承，不重新训练。

## 7.1 已知残余混杂

- Control 的 `legacy` recurrent 会处理 padding；候选 Transformer 使用显式 padding
  mask。这是整体 backbone 替换带来的语义差异。既有 paired Packed-control 没有形成
  稳定提升，因此不为本轮重复训练，但报告中必须披露。
- MOSI 只有 1 个 speaker。其 speaker embedding 对有效 utterance 是常量条件，不能
  据此声称模型学习了 speaker interaction。
- 参数预算近似相等不代表 FLOPs 相等；必须记录单 batch wall time 与峰值显存。

## 8. 测试

实现前先写以下失败测试：

1. 输出形状、padding 置零和 device/dtype。
2. padding-value invariance。
3. future utterance 改变能够影响 earlier hidden，证明是 full-context 而非 causal。
4. speaker id 改变能够影响输出；padding speaker embedding 不泄漏。
5. 所有 Transformer 层、输入投影、speaker embedding 有效行、final norm 与输出
   投影均获得有限非零梯度；speaker padding row 梯度严格为零。
6. registered 参数为 `5,869,754`，effective active-forward 参数为 `5,869,370`，
   后者与 `5,864,700` 的差异小于 `0.2%`。
7. 新模型保持现有 Missing-M3 tuple、teacher update 和 predictor target mask 合同。
8. 同 seed 初始化与 forward 确定；不同 seed 初始化不同。
9. 同 seed 下所有共享模块的初始 tensor 与 Control 精确相同。
10. CPU FP32 完整 forward/backward；远程 CUDA FP32 单 batch forward/backward。
11. Runner dry-run 生成且只生成 seeds 66–70，GPU 仅使用 0、1、2，Original 不在命令中。
12. Result manifest 记录 commit、环境、配置 SHA、mask SHA、参数口径和完成状态。
13. 单 batch 记录 candidate 的 forward/backward wall time 与 peak GPU memory，并与
    同 batch Control 做一次只读 profiling；该 profiling 不训练新的 Original。

不增加重复的 1-epoch checkpoint/smoke 训练。单元测试与一次 CUDA batch 验证通过后，
直接启动正式任务。

## 9. 正式实验与判定

运行 5 个模型：seeds 66、67、68、69、70。每个模型使用同一个 mixed-rate 训练协议，
并在最佳 validation checkpoint 上测试 8 个 rates。

预注册主判据：

- 五种子 validation 8-rate mean 至少比 strict control 的 `78.7675` 高
  `0.50` 个百分点；
- 至少 4/5 seeds 的 validation delta 为正；
- high-missing validation mean 不下降；
- miss-0 validation delta 不低于 `-0.30`；
- 无单 sign、常量输出、非有限 loss 或有效 rank 异常坍塌。

测试结果无论正负都完整归档，但不能反向用于改动本候选。若未通过，结论为
「GCNet 整体替换仍不足以突破当前 frozen-feature 上限」，随后才转入独立 upstream
lane；不得根据 test 继续微调本 Transformer。

## 10. GitHub 交付

完成后执行：

1. 更新新目录 `STATUS.md` 与 `results/SUMMARY.md`；
2. 运行目标测试、相关回归测试、`py_compile` 和 `git diff --check`；
3. 使用 Lore Commit Protocol 提交；
4. 推送到 `github/feature/m3-jepa-gcnet`；
5. 在最终报告中给出 commit、目录、五种子逐 seed 分数、八 rate 均值和失败风险。
