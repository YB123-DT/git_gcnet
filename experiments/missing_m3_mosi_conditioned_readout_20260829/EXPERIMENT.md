# MOSI Availability-Conditioned Readout Study

## 研究问题

现有 Slot Missing-M3 已把 availability pattern 编入图前节点，但图后所有
`A/T/V/AT/AV/TV/ATV` utterance 仍共用同一个线性 MOSI 回归头。既有实验已经基本关闭：

- 扩大 hidden/window；
- time attention；
- Temporal-only / Speaker-only；
- test-time missing completion；
- local/context residual、modality track、PCIR；
- 更复杂或 paper-faithful M3 MMoE。

因此本轮只检验一个尚未回答的问题：不同 availability 条件下的图表示是否需要共享但不完全相同的最终决策映射。

## 唯一改动

保留原共享头：

```text
base = W0 h + b0
```

增加 rank-8 pattern-conditioned residual：

```text
u = U LayerNorm_no_affine(h)
delta = contract(u, V[availability]) + bias[availability]
prediction = base + delta
```

七种有效 pattern 共用 `U`，只有低秩输出因子和 bias 按当前 utterance 的真实 mask 路由。`U` 使用标准初始化，`V/bias` 为零初始化，因此训练前与 Shared 完全等价；ATV 在训练后允许学习自己的残差，不强制恢复 Original。

正式 MOSI 配置的 hidden 为 `250`、rank 为 `8`、输出为 `1`，新增参数：

```text
250*8 + 7*8*1 + 7*1 = 2063
```

不修改 frozen wav2vec/DeBERTa/MANet features、Slot encoder、Temporal/Speaker graph、Missing-M3 predictor、JEPA loss 或训练 rate 协议。

## 锁定协议

```text
dataset           CMU-MOSI
checkpoint        one unified model for all eight rates
train rates       0.0--0.7, all-rates-per-batch
seeds             66, 67, 68
hidden            100
graph hidden      50
window            1/1
time attention    false
fusion            slot
graph branches    both
learning rate     5e-4
weight decay      1e-5
JEPA weight       0.1
readout rank      8
```

Shared controls直接继承 hidden/window sweep 的相同三种子结果，不启动 Original/Shared 训练。

## 验证集门槛

选择只读取 `history.json`，不读取 `metrics.json` 或 prediction NPZ：

- 三种子 validation 八 rate mean delta `>= +0.40` percentage points；
- 至少 `2/3` seeds 为正；
- high missing `0.4--0.7` mean delta 非负；
- miss-0 delta `>= -0.30` point；
- 任一 seed overall delta 不低于 `-1.00` point；
- 无非有限值、常量预测、单一 sign 或 validation W-F1 `<= 0.55` 的坍塌。

只有通过该门槛才扩展 seeds `69,70`。Screen 显式使用 `--skip-test-evaluation`：门槛决定前不计算 test、不生成 prediction NPZ，并保留 validation-selected checkpoint 供通过后单独评估。Test 不能选择方案、rank、epoch 或触发调参。

## TDD 与环境证据

- 本地相关回归：`105 passed`；
- biggpu 固定 `s0` 环境：`105 passed`；
- biggpu `gcnet-official`、GPU7 FP32 forward/backward：通过；
- Shared 模式不实例化新参数，state keys、RNG 初始化、forward 和 strict checkpoint loading 保持原样；
- runner 禁止 GPU4；screen 保留 checkpoint，完成 validation gate 与可复现性审计后再决定清理。
- 初次 provisional 队列在约 epoch 30 被主动停止：代码审查发现新增层会推进 CPU RNG，且旧 NPZ availability 的 `[L,B]` 顺序与 prediction 的 `[B,L]` 顺序不一致。正式队列在 RNG 隔离、alignment 回归测试和 strict validation-only mode 通过后重新开始；provisional 文件不参与结果。

## 路径

- Remote candidate root：`/data2/yb/remote_experiments/missing_m3_mosi_conditioned_readout_20260829/results/availability-low-rank_rank8`；
- Remote inherited control：`/data2/yb/remote_experiments/missing_m3_mosi_hidden_window_sweep_20260829`；
- Runner：`scripts/run_mosi_conditioned_readout.py`。

## 状态

`COMPLETE — FAIL at validation gate`。

## 三种子 Validation 结果

| Seed | Shared | ACLR | Delta |
|---:|---:|---:|---:|
| 66 | 78.4533 | 76.9696 | -1.4837 |
| 67 | 79.0164 | 79.5564 | +0.5400 |
| 68 | 78.0300 | 77.8773 | -0.1527 |
| **Mean** | **78.4999** | **78.1344** | **-0.3655** |

其他预注册门槛：

- positive seeds：`1/3`，要求 `>=2/3`；
- high missing delta：`-0.8061` point，要求非负；
- miss-0 delta：`-0.6016` point，要求 `>=-0.30`；
- collapse：无。

因此 ACLR 被关闭，不扩展 seeds 69/70、不运行 parameter-matched control、不计算 test。下一步严格按 PRD 进入 Option B：zero-initialized availability-conditional affine residual。

## Option B：Availability-Conditional Affine Residual

Option A 已失败，因此按预注册顺序仅替换最终读出机制：

```text
n = LayerNorm_no_affine(h)
h_conditioned = h + gamma[availability] * n + beta[availability]
prediction = W0 h_conditioned + b0
```

`gamma/beta` 只包含七个有效 availability 组合；padding 在路由前排除，
不分配可学习行。两张表均为零初始化，所以固定共享参数下初始输出与
Shared 完全一致。该残差只作用于最终情绪读出，返回的 GCNet hidden、
Missing-M3 predictor、JEPA、EMA teacher 和图传播均不改变。

正式 hidden 为 `250`，新增参数为：

```text
2 * 7 * 250 = 3500
```

执行协议、三种子 validation 门槛和 inherited Shared controls 与 Option A
完全相同；screen 仍不计算 test。

验证证据：

- 本地相关回归：`112 passed`；
- biggpu 固定 `s0` 环境：`112 passed`；
- biggpu `gcnet-official`、GPU7 FP32 forward/backward：通过；
- 七 pattern 路由、padding 隔离、命中行梯度、共享初始化/RNG/初始输出
  等价、最终读出边界和参数预算均已锁定。

状态：`COMPLETE — FAIL at validation gate`。

### 三种子 Validation 结果

| Seed | Shared | Affine | Delta |
|---:|---:|---:|---:|
| 66 | 78.4533 | 77.7114 | -0.7420 |
| 67 | 79.0164 | 78.2977 | -0.7187 |
| 68 | 78.0300 | 77.6125 | -0.4176 |
| **Mean** | **78.4999** | **77.8738** | **-0.6261** |

其他门槛：

- positive seeds：`0/3`；
- high-missing delta：`-0.7513` point；
- miss-0 delta：`-0.4742` point；
- collapse：无；
- selection source：`validation-only`，未计算 test。

因此 Option B 被关闭，不扩展 seeds 69/70。两种读出改造均未改善结果，
下一步按预注册顺序进入 Option C：仅改变 JEPA regression 的聚合单位，
不增加模型参数或推理路径。

## Option C：Utterance-Balanced JEPA Regression

现有 JEPA regression 把所有实际缺失 target 直接拼接后求均值。因此：

- `A/T/V` pattern 每句话有两个缺失 target；
- `AT/AV/TV` pattern 每句话只有一个缺失 target；
- 前一类 utterance 在 regression 中天然获得两倍权重。

Option C 只把聚合改为：先在每个 utterance 的真实缺失 targets 内求均值，
再对有 target 的 utterances 求均值。target-specific InfoNCE、JEPA 权重 `0.1`、
分类损失、EMA teacher、模型参数和推理路径均保持不变。默认
`jepa_regression_aggregation=target` 保留旧行为；候选显式使用 `utterance`。

TDD 已锁定：单 target 等价、复制相同 target 误差不改变 utterance 权重、
默认旧聚合精确等价、零 target 可有限反传。当前本地相关回归：
`117 passed`。

状态：`COMPLETE — FAIL at validation gate`。

### 三种子 Validation 结果

| Seed | Target-balanced Shared | Utterance-balanced | Delta |
|---:|---:|---:|---:|
| 66 | 78.4533 | 77.8235 | -0.6299 |
| 67 | 79.0164 | 79.3890 | +0.3726 |
| 68 | 78.0300 | 77.0302 | -0.9998 |
| **Mean** | **78.4999** | **78.0809** | **-0.4190** |

其他门槛：

- positive seeds：`1/3`；
- high-missing delta：`-0.4498` point；
- miss-0 delta：`-0.8123` point；
- collapse：无；
- selection source：`validation-only`，未计算 test。

因此保留原 target-balanced JEPA regression。单源 utterance 的两个真实缺失
targets 获得两份监督在当前模型中是有效的困难样本加权，而不是需要修正的偏差。
Option C 不扩展 seeds 69/70。

## Option D：Length-Aware Packed Recurrent Path

Options A--C 关闭后，代码级审计发现 Original/当前 GCNet 的图前 BiLSTM 和
Temporal/Speaker 两个图后 BiLSTM 都直接读取补零序列，未使用已经传入的
`seq_lengths`。由于它们是双向、两层 recurrent，同一句有效 utterance 的输出会
随 batch 中最长对话和人工 suffix padding 变化。固定随机 LSTM 的独立诊断中，
同一 9-step 序列补零到 63 后有效输出最大变化约 `0.095`。

唯一模型变量：

```text
legacy: recurrent(padded tensor)
packed: pack(valid lengths) -> recurrent -> unpack(total length)
```

该开关同时作用于图前 recurrent 和两个图后 recurrent，新增参数为 `0`；GCN、
图拓扑、Slot、JEPA、task loss、head、mask 与 checkpoint 选择不变。默认
`legacy` recurrent 分支保留旧 checkpoint/state-key/RNG/forward 计算；但旧结果的
relation ID 语义未知，因此只保证 checkpoint 严格加载，不把历史数值当本轮 control。

### 一次性可复现性修复

审计同时发现 `batch_graphify` 通过 Python `set` 枚举 relation ID。实测不同
`PYTHONHASHSEED` 会改变 `past/now/future` 与 speaker relation 的编号，历史结果
没有保存该映射，不能作为严格配对 control。因此本轮固定：

```text
temporal: past=0, now=1, future=2
speaker: 00=0, 01=1, 10=2, 11=3
PYTHONHASHSEED=0
```

并一次性新跑 seeds 66--68 的 deterministic legacy 与 packed。此后候选都继承
这批 deterministic legacy，不再重复 Original。

验证证据：

- 本地完整相关回归：`124 passed`；
- biggpu 固定 `s0`：`124 passed`；
- relation mapping 在四个不同 Python hash seed 下完全一致；
- 默认/显式 legacy 的 state keys、参数、RNG 和 strict checkpoint loading 一致；
- equal-length legacy/packed 输出等价；
- packed 下短对话单独运行或与长对话同 batch 时有效输出一致；
- 图前、Temporal-post、Speaker-post 三处均经过 packed helper；
- 真实 MOSI GPU7、1 epoch、all-rates forward/backward/EMA 完成，参数量不变，
  test 为空。

正式输出：

```text
results/packed-recurrent/seed_{66,67,68}
results/deterministic-legacy/seed_{66,67,68}
```

状态：`COMPLETE — FAIL at validation gate`。

### 三种子 Validation 结果

| Seed | Deterministic Legacy | Packed | Delta |
|---:|---:|---:|---:|
| 66 | 77.3096 | 78.5192 | +1.2096 |
| 67 | 78.6790 | 79.3712 | +0.6922 |
| 68 | 78.2220 | 76.5922 | -1.6298 |
| **Mean** | **78.0702** | **78.1609** | **+0.0907** |

其他门槛：

- positive seeds：`2/3`；
- high missing delta：`-0.3432` point；
- miss-0 delta：`+0.6243` point；
- worst seed delta：`-1.6298` points；
- collapse：无；
- selection source：`validation-only`，未计算 test。

Packed path 虽修正了人工 suffix padding 对双向 recurrent 的污染，但提升远低于
`+0.40` point 门槛，且 high-missing 与 worst-seed 保护项失败。因此它只保留为
可选的正确性模式，不作为性能主方案，也不与下一候选捆绑。新鲜的
`deterministic-legacy` 三种子结果成为后续单变量候选的锁定对照，不再重训。

下一项是 MOSI emotion task regression 的鲁棒损失筛选：只把 MSE 换为
`SmoothL1(beta=1.0)`，保持 Legacy recurrent、Shared readout、target-balanced JEPA、
八 rate mixed training 与所有模型参数不变。

## Option E：Robust Emotion Regression Objective

MOSI 的连续标签位于 `[-3,3]`。当前 all-rates-per-batch 协议让同一 utterance 在
八种缺失 view 中重复贡献主任务梯度；MSE 会让少量大残差在每个 view 中都获得
二次权重。Option E 只将 emotion task loss 改为：

```text
MSE(valid continuous labels)
→ SmoothL1(valid continuous labels, beta=1.0)
```

以下均不改变：

- 零标签继续参与连续训练 loss；
- W-F1/Acc-2 计算继续排除零标签；
- MAE/correlation 继续使用全部有效连续标签；
- JEPA loss、EMA、mask schedule、八 rate 权重、模型参数和推理路径；
- checkpoint 仍由 validation 八 rate mean W-F1 选择。

默认 `task_regression_loss=mse` 已通过 loss 与 gradient 的逐元素精确等价测试；
SmoothL1 CUDA FP32 forward/backward 有限。正式候选固定 `beta=1.0`，不进行 beta
sweep，并复用 Option D 产生的 deterministic Legacy MSE controls。

解释边界：标准 SmoothL1 除了限制大残差梯度，也改变了主任务与 `0.1×JEPA` 的
相对梯度尺度。因此若首轮通过，只能先声称“SmoothL1 objective 有效”；若要进一步
归因于 tail robustness，之后需要单独做 loss-scale control，不能直接作机制结论。

状态：`COMPLETE — FAIL at validation gate`。

### 三种子 Validation 结果

| Seed | MSE Control | SmoothL1 | Delta |
|---:|---:|---:|---:|
| 66 | 77.3096 | 78.7596 | +1.4499 |
| 67 | 78.6790 | 78.3977 | -0.2813 |
| 68 | 78.2220 | 77.7924 | -0.4296 |
| **Mean** | **78.0702** | **78.3166** | **+0.2464** |

其他门槛：

- positive seeds：`1/3`；
- high missing delta：`-0.2486` point；
- miss-0 delta：`+0.4860` point；
- worst seed delta：`-0.4296` point；
- collapse：无；
- selection source：`validation-only`，未计算 test。

SmoothL1 显示弱正向均值，但未达到 `+0.40` point，且 positive-seed 与
high-missing 保护项失败。因此不扩展 seeds 69/70、不查看 test、不扫描 beta，也不与
下一模型候选组合。

## Option F：Shared Post-Graph BiLSTM

Fresh Legacy 在约 epoch 45--50 达到验证最佳，之后训练 W-F1 继续升至约 94%，
而验证 Val8 降至约 71--72%，显示明显过拟合。hidden=100 模型中，Temporal 与
Speaker 两支图后 BiLSTM 各含 `2,508,000` 参数，合计接近当前可训练参数的一半。
既有单分支消融又证明两张图都有贡献，因此 Option F 不删除任何图，只共享重复的
sequence dynamics：

```text
Temporal RGCN/GraphConv -> shared post-BiLSTM -> Temporal linear -> ReLU
Speaker  RGCN/GraphConv -> shared post-BiLSTM -> Speaker  linear -> ReLU
                                                        |
                                                  Original addition
```

唯一开关：

```text
postgraph_sequence_mode = independent | shared-bilstm
```

两套 RGCN/GraphConv 和 branch-specific linear 保持独立；所有模块仍按 Legacy 顺序
实例化，以保持初始化 RNG 和 state keys。candidate 使用 Temporal branch 的
`grufusion` 处理两路 graph sequence，Speaker branch 的 `grufusion` 保留 checkpoint
keys 但冻结且不执行。预计减少 `2,508,000` 个可训练参数，不改变两张图、两次图后
sequence forward、MSE、JEPA、mask 或推理输出接口。

正式 screen 继续复用 deterministic Legacy MSE controls，SmoothL1 恢复为 MSE；
通过门槛与前述选项完全相同。若失败，不再尝试 shared-linear、部分层共享、冻结比例
或与 SmoothL1 组合。

### Option F 三种子筛选结果

三种子状态曾为 `SCREEN PASS`。全部任务完成 100 epochs，按 validation 八个 missing rates 的
mean W-F1 选择最佳 epoch，未读取 test：

| Seed | Shared BiLSTM | Deterministic Legacy | Delta (point) | Best epoch |
|---:|---:|---:|---:|---:|
| 66 | 78.6313 | 77.3096 | +1.3217 | 50 |
| 67 | 78.6608 | 78.6790 | -0.0182 | 47 |
| 68 | 78.5218 | 78.2220 | +0.2998 | 43 |
| Mean | 78.6046 | 78.0702 | +0.5344 | -- |

保护项：high-missing `+0.6239` point，miss-0 `+0.7642` point，`2/3`
seeds 为正，无坍塌。三组 candidate/control 的 37 个非 treatment 配置字段一致。
这是本轮第一个通过预注册三种子门槛的候选。

随后仅补了 seeds 69、70 的 Shared BiLSTM 与对应 deterministic Legacy controls；
seeds 66--68 未重跑。

### Option F 五种子确认结果

最终状态：`COMPLETE — FAIL at five-seed validation gate`。

| Seed | Shared BiLSTM | Deterministic Legacy | Delta (point) |
|---:|---:|---:|---:|
| 66 | 78.6313 | 77.3096 | +1.3217 |
| 67 | 78.6608 | 78.6790 | -0.0182 |
| 68 | 78.5218 | 78.2220 | +0.2998 |
| 69 | 79.2124 | 81.5401 | -2.3277 |
| 70 | 77.9187 | 78.0868 | -0.1681 |
| Mean | 78.5906 | 78.7691 | -0.1785 |

五种子保护项：positive seeds `2/5`，high-missing `-0.6508` point，miss-0
`+0.5693` point，无坍塌。所有 candidate/control 配置审计通过，config/history
SHA256 已写入 `FIVE_SEED_VALIDATION_SUMMARY.json`。

结论：硬共享图后 BiLSTM 在部分初始化下可降低过拟合并改善 complete 条件，但会
压低强 Legacy 初始化的上限，且总体伤害 high-missing。三种子正向属于不稳定信号，
该路线不计算 test，不尝试 partial sharing、shared-linear、冻结比例或与 SmoothL1
组合。

## Option G：Active-Rate Coefficient-Budget Weighted JEPA

已有五种子消融显示 Uniform JEPA 对 overall/high-missing 分别贡献
`+0.4559/+0.6593` point，均为 `4/5` seeds 正向；收益主要出现在中高缺失率。
Option G 保持 target-balanced JEPA 内部公式不变，只在 all-rates-per-batch 聚合处使用：

```text
loss_eta = task_eta + 0.1 * ((1 + eta) / 1.4) * jepa_eta
```

`eta=0` 没有 missing target；`0.1--0.7` 七个 active rates 的系数均值精确为 1。
因此它是温和的 rate 间名义系数预算重分配，不是增加总 JEPA 权重，也不声称理论最优
或实际梯度范数守恒。模型、参数、mask、forward、推理和 checkpoint rule 全部不变。

正式 screen 只运行 seeds 66--68、validation-only，复用 direct deterministic Legacy
controls。若未通过原门槛，不扫描其他曲线，也不与其他 treatment 组合。

### Option G 三种子结果

状态：`COMPLETE — FAIL at validation gate`。

| Seed | Weighted JEPA | Uniform control | Delta (point) | Best epoch |
|---:|---:|---:|---:|---:|
| 66 | 77.5610 | 77.3096 | +0.2513 | 54 |
| 67 | 78.2950 | 78.6790 | -0.3840 | 45 |
| 68 | 77.5969 | 78.2220 | -0.6251 | 45 |
| Mean | 77.8175 | 78.0702 | -0.2526 | -- |

high-missing delta `-0.5838` point，miss-0 delta `+0.0031` point，仅 `1/3`
seeds 为正，无坍塌，37 个非 treatment 配置字段一致。结果说明 rate-level 加权同时
放大了高缺失 target 的歧义噪声；“JEPA 有效”不能推出“按 rate 增强 JEPA 更有效”。
不扩 seeds 69/70、不读取 test、不扫描其他权重。

## Option H：Branch-Specific Graph-Message Calibration

Shared BiLSTM 的失败说明两支 graph→sequence dynamics 不能硬共享。Option H 保留两套
RGCN/GraphConv 与两套图后 BiLSTM，只在每支 `conv2` 输出进入 BiLSTM 前加入：

```text
m'_b = m_b + tanh(alpha_b) * (LayerNorm_no_affine(m_b) - m_b)
```

Temporal/Speaker 各有独立的 `[D_g]` 零初始化 `alpha_b`。正式 hidden100 配置只增加
`2*50=100` 个参数；默认 `none` 不实例化新 key，treatment 初始 graph message 与
control 精确一致。Uniform JEPA、MSE、Legacy recurrent、independent BiLSTM、mask、
graph topology 与推理均保持不变。

正式 screen 只运行 seeds 66--68、validation-only，并继承 direct deterministic
Legacy controls；通过 gate 前不读取 test。

### Option H 三种子结果

状态：`COMPLETE — FAIL at validation gate`。

| Seed | Calibrated | Raw-message control | Delta (point) | Best epoch |
|---:|---:|---:|---:|---:|
| 66 | 77.7295 | 77.3096 | +0.4199 | 44 |
| 67 | 78.7708 | 78.6790 | +0.0919 | 51 |
| 68 | 77.8963 | 78.2220 | -0.3257 | 47 |
| Mean | 78.1322 | 78.0702 | +0.0620 | -- |

high-missing delta `-0.4704` point，miss-0 delta `+1.0748` points，`2/3`
seeds 正向，无坍塌，37 个非 treatment 配置字段一致。该校准能改善 complete 条件，
但仍以牺牲高缺失为代价，未达到 overall `+0.40` point 与 high-missing 保护项。
因此不扩 69/70、不读取 test，也不扫描 normalization/alpha/branch-only 变体。

三组最佳 checkpoint 的 Temporal/Speaker `alpha` 均为非零，证明 treatment 路径实际
参与训练；这仅用于排除“开关未生效”，不解释为模型主动接受或拒绝归一化。G/H 的
artifact SHA256、source provenance 与历史 manifest 的 MOSI 单-speaker relation 映射
校正见 `PROVENANCE_AUDIT.json`。历史 manifest 保持不可变，校正不影响数值结果：MOSI
实际只有 `00 -> 0` 一种 speaker relation。
