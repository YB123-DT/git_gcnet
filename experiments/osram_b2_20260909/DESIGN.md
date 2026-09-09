# B2: Source-only Pre-OSRAM Completion

状态：实现与单批次验收完成；**没有运行完整 Stage1 / Stage2 训练，没有 B2 F1 结果。**
分支：`feature/osram-complete`。基于 `fca2aa3`，保留全部旧模式。

## 数据流

Stage1（固定空间预训练）：

```text
当前真实 observed raw → frozen Student Projectors → SourceOnlyM3Predictor
                                                       ↓ reg / cl predictions
完整 target raw → frozen EMA Teacher projectors → existing missing_m3_loss
```

不实例化/运行 OSRAM、emotion classifier 或 contextual predictor。SourceOnlyPretrainer 仅包含冻结的 projectors/teacher 和新的 source-only MMoE。两套 frozen bank 都处于 eval，所有参数 requires_grad=False；Stage1 不更新 EMA。只对新 predictor 使用原 Adam/LR/weight-decay/clip 规则。沿用 cyclic natural-mask schedule；complete batch 没有缺失 target 时跳过 predictor optimizer step。

Stage2（一次联合前向）：

```text
incomplete → 原 ObservedSetEncoder → observed_node + real observed latents
                                      │                    ↓
                                      │         SourceOnlyM3Predictor
                                      │                    ↓
                                      │         filled A/T/V + source status
                                      │                    ↓
                                      │          CompletedReadFusion
                                      │                    ↓
                              write_node=observed     read_node=completed
                                      └──────────┬─────────┘
                                             OSRAM ×1
                                                ↓
                                           原 emotion head
```

总损失保持 `L_emotion + config.jepa_weight * existing missing_m3_loss`。reg_predictions 直接受到 SmoothL1 target supervision；contrastive 仍按原配置使用 cl/reg 分支，未重新设计损失。Stage2 projector、source predictor、OSRAM 和分类头正常学习；teacher 没有梯度，optimizer step 后按原 tau 更新 EMA。旧 ContextualM3Predictor 保留 state keys，但 B2 下不调用且不加入可训练参数。

测试保留 source-only predictor + read fusion + OSRAM，完全不调用 Teacher。分类路径与训练一致。预测 raw missing feature 不进入 source；complete raw 只进 teacher loss 分支。

## Source-only predictor / fusion

- `SourceOnlyM3Predictor.forward(latents, availability, umask)` 没有任何 hidden/context/history 参数。
- 重用 DualGateTopKMMoE 的原 source/target embeddings、experts、routing、reg/cl heads。
- 固定遍历 A/T/V，对每个真正 missing target，只取当前真实 observed source。
- 例如 AV→T，`mean(reg(A→T), reg(V→T))`；source mean 不包含 missing/target/padding。
- 返回原 MissingM3Predictions 类型 `[L,B,3,d]`、target mask、source counts；reg 是唯一回灌分支。
- Filled slots 使用 `where(observed, real_latent, reg_prediction)`；真实槽原样保留。
- 每槽 `LayerNorm(latent)` + `[1,0] observed / [0,1] predicted`；身份不是置信度。
- A/T/V 固定 concat，LN→Linear→GELU→Dropout→Linear(d)，末层 weight/bias 全零。
- residual 仅计算有效且不完整 utterance；ATV 精确保持 observed_node，padding residual 严格零。

**零初始化梯度语义：** 初始分类 loss 对 predictor 的梯度为零，但融合末层已有非零梯度。首步更新后分类梯度可以进入 predictor。这是规格中零初始化的必然结果，不能虚称第一步即非零。

## OSRAM read/write 的准确边界

| 路径 | 输入 | B2 prediction 能否影响 |
|---|---|---|
| Local / local skip | read_node | 能 |
| Base / Gap query | LN(read_node), 原 availability, speaker, query type | 能 |
| Key | real latent, LN(write_node=observed_node), 原 availability, speaker | 不能 |
| Value | real observed latent | 不能 |
| Gap residual operator 的 K_obs | 上述 write-side keys | 不能 |
| decay / beta / ridge / block correction | 原真实写入路径 | 不能 |

不改 alpha、beta、ridge、read-before-write、masked mean、Gap R、write-step .6、heads/key/value 维度。没有 second OSRAM，没有预测槽写入，没有 attention、confidence 或新 loss。

### 核心验收

**在固定参数、相同真实输入/availability/初始状态、确定性 eval 条件下，normal / zero / random / same-pattern-shuffled predictions 对应的每个时间步 persistent memory 逐元素相同。**

测试截获每次 block_write 的 pre-state、keys、values、post-state 并用 torch.equal 比较；同时验证 Query/Local/readout 可以变化。padding masking 不变，因此有效状态轨迹也相同。旧 scan/update 完全没改。

这不声称“B2 训练前后 memory 相同”：Stage2 的新梯度会学习 shared projector/key/value 参数；不同训练后权重当然可以产生不同轨迹。训练态随机 dropout 对照应固定相同随机性，不能拿独立随机前向作精确一致断言。

## checkpoint 转移与兼容

- 默认 `completion_path=none`。不创建新 B2 参数，旧 config 缺字段用 dataclass 默认，旧 state 可 strict load；历史 default OSRAM 参数、RNG、输出、梯度均精确比较通过。
- 旧 `classification_completion` 原语义保留。与 `pre_osram_b2` 同开直接 ValueError。
- B2 第一版只接受 full、mean、causal、无 slot reuse、write_step=.6；其他 OSRAM 模式原样保留供旧实验使用。
- Stage1 从当前 causal .6 base checkpoint 复制 frozen Student/Teacher，**新 source-only predictor 随机初始化**，不是偷偷复用 contextual context projection。
- Stage1 保存 `source_only_completion_pretrain.pt`：predictor、frozen bank tensors、base 文件 SHA256、frozen space SHA256、配置、dimensions、epoch、aggregation 版本、feedback branch、optimizer state、torch RNG。
- Stage2 base 文件 SHA256 必须与 Stage1 相同；frozen bank 实际 tensor hashes 必须一致；Stage1/source config 必须一致。所有 shared keys 都必须存在，唯一允许新增 keys 是 `source_only_predictor.*` / `completed_read_fusion.*`。
- 非 state 配置如 top-k、ridge、dropout、query availability 也核对，不能只靠 strict shape 检查。
- Stage2 从两个 checkpoint **初始化新的 optimizer**，不是精确恢复旧训练；不恢复旧 optimizer/scheduler 轨迹。Teacher 初值来自原 checkpoint，后续沿用原 EMA。

## 后续 control 接口（未运行）

`completion_predictions_override: Tensor[L,B,3,d]` 仅 eval 时允许，覆盖分类所用 reg latent。实际 predictor output / target mask 保留不变。可据此注入训练集 prototype 或 same-pattern/target shuffle，不改融合层和 mask。

Stage2 复用原 evaluate_rate 的 `predictions_miss_*.npz`，保存 aligned predictions、labels、availability，可按七 pattern 及 NO_TEXT / TEXT_PRESENT 分组。没有新增 pattern 特定 eta。

## 已验证参数量（当前真实 MOSI 配置）

| 项目 | 参数数 |
|---|---:|
| 新 SourceOnlyM3Predictor | 925,192 |
| 新 CompletedReadFusion | 266,252 |
| 新增总数 | **1,191,444** |
| B2 全部参数（含冻结 Teacher、保留旧 predictor） | 8,372,661 |
| Stage2 可训练参数 | 5,793,965 |

没有增添 OSRAM memory 参数。新参数不等于必然提升；本轮不做参数匹配训练。

## 证据与限制

- [SMOKE.json](SMOKE.json)：真实 MOSI seed66，32 conversations / 843 valid utterances。
- 一个真实 batch：Stage1 一个 optimizer step；为验证 zero-init 解锁，Stage2 在同一 batch 上两个 optimizer steps；不是完整 epoch。
- Stage1 regression loss .430536、total 3.448107，predictor gradient norm 3.804715。
- Stage2 emotion→predictor gradient：step1=0，step2=2.598421；只证明梯度链路，不是训练效果。
- Stage2 每个 forward 只有一次 OSRAM；test Teacher calls=0；保存/strict reload 后 eval logits 精确相同。
- 所有数字是 smoke 数据，**不可作为预训练质量或 F1 结论**。
- 目前没有完整训练后的 B2 checkpoint、Stage1 latent audit 结论或 P1 结果。Stage1 完整 runner 训练结束会生成 `latent_audit.json`（miss=.5 A/V/AV，reg/cl 分开），只分析不选模型。
- 基础 checkpoint 已经经过 Test-oracle selection；未来由它初始化的结果不能自动宣称独立正式验证。
