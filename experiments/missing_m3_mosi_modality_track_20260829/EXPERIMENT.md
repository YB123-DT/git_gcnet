# CMU-MOSI Modality-Track GCNet

## 研究问题

当前 Slot Missing-M3 在图前将 observed A/T/V 压缩为一个 node。本实验检验：让三种模态
分别经过同一套共享参数的 GCNet，并把跨模态融合延迟到图后，能否改善 mixed-rate
checkpoint 的统一缺失鲁棒性。

## 唯一方法变量

- Control（继承）：图前 Slot Fusion，一次 GCNet；
- Treatment：A/T/V 三条 track 顺序调用同一个 `encode_hidden`，图后按 availability
  执行 Slot Fusion；
- 三条 track 共享 pre-graph recurrent、Temporal/Speaker graph 与 post-graph recurrent
  参数，不是三个 backbone；
- Missing-M3 Predictor、EMA Teacher、loss、mask、all-rates-per-batch、classifier、图拓扑
  和测试协议不变；
- Predictor/Teacher 仍只在训练期存在，测试不生成或回灌缺失模态。

正式对照中，Control 与 Treatment 的所有 shared state key 在相同 seed 下逐 tensor
初始化一致；二者融合 dropout 均为 0.1。第一次仅运行至约 3--4 epoch 的队列因发现
dropout/RNG 配对混杂而作废并清理，不计入结果。

## 协议

- 数据集：CMU-MOSI official split，fold 1；
- 任务：Regression-MSE，报告 non-zero sentiment Acc-2/W-F1；
- 训练：一个 all-rates-per-batch checkpoint 测试 `0.0,...,0.7`；
- seeds：66、67、68、69、70；
- hidden=200，latent=256，window=2/2，time-attn=False；
- 100 epochs；按验证集八个 rates 的 W-F1 均值选唯一 checkpoint；
- Control 五种子直接继承，不重跑；40/40 test mask SHA256 必须配对。

## 实现与验证

- TDD 红灯分别证明缺少 track encoder/fusion、model switch 和 CLI；
- 修复审查发现的两个正式混杂后，远程完整测试 `90 passed`；
- 测试覆盖七 pattern、raw missing-value 泄漏、padding、共享图三次调用、默认路径 key
  等价、shared initialization、CLI、完整 backward 与端到端 logits 泄漏；
- official 环境 1-epoch smoke 完成，8 个 rate 均可验证/测试；
- 40/40 NPZ 的 W-F1 已从 `predictions`/`labels` 独立重算，与 `metrics.json` 一致；
- 40/40 mask SHA256 与 paired Control 一致。

## 五种子结果

| Miss | Modality-Track | Slot Control | Delta | 正向 seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 83.47 | 85.76 | -2.28 | 1/5 |
| 0.1 | 79.94 | 83.05 | -3.11 | 0/5 |
| 0.2 | 76.87 | 80.79 | -3.92 | 0/5 |
| 0.3 | 74.75 | 79.20 | -4.45 | 0/5 |
| 0.4 | 71.36 | 76.20 | -4.84 | 0/5 |
| 0.5 | 68.76 | 74.80 | -6.04 | 0/5 |
| 0.6 | 65.70 | 73.30 | -7.60 | 0/5 |
| 0.7 | 65.43 | 71.37 | -5.93 | 0/5 |

- 八-rate 均值：73.286 vs 78.059，delta=-4.773；
- 高缺失 0.4--0.7：67.812 vs 73.916，delta=-6.104；
- 总体正向 seed：0/5；高缺失正向 seed：0/5；
- miss0 也下降 2.284，未满足“不低于 -0.3”的门槛；
- 最佳 epochs：98、88、51、78、89；没有最终类别常量坍塌；
- 参数量：32,831,737，Control 为 32,089,733，新增 742,004；
- 五个并行任务各约 15.5--16.7 分钟；同机 seed66 Control 历史记录约 7.4 分钟，说明
  图计算 wall time 约增至 2.2 倍（不同并发时段，仅作运行成本估计）。

## 结论

`COMPLETE — FAIL`。

延迟融合在所有 nonzero missing rates 和所有 seed aggregate 上均下降，且缺失率越高，
平均损失越大。证据不支持“图前融合是当前主要瓶颈”。更符合结果的解释是：GCNet 的
Temporal/Speaker propagation 需要每个 utterance 的联合多模态证据；把模态拆成稀疏
track 后，高缺失条件下每条图的有效当前节点更少，跨模态互补只能在传播结束后发生，
已经无法修复图内上下文推理的信息缺口。

本路线到此关闭，不追加 attention、可靠性 gate 或独立 backbone 来挽救，否则会改变
研究问题并失去单变量归因。原 Slot training-only DualGate 继续作为当前最佳版本。

## 位置

- 远程 checkpoint：`/data2/yb/remote_experiments/missing_m3_mosi_modality_track_20260829/formal`；
- Git 仅保存轻量 config、history、metrics 与 40 个 prediction NPZ，不提交 `best.pt`。
