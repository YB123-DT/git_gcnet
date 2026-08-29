# CMU-MOSI PCIR 图前联合节点构造实验

## 研究问题

现有 Slot node 保留 Audio、Text、Visual 的可见模态身份，但图前没有显式表示两个可见
模态之间的一致、差异和乘性交互。本实验只增加 **Pattern-Conditioned Interaction
Residual (PCIR)**，判断显式 observed-pair interaction 是否能改善 Missing-M3 + GCNet。

PCIR 不预测或回填缺失模态。它先按 availability 将 missing Student latent 严格清零，
再计算：

```text
observed unary = pattern/modality-specific scale-shift(student latent)
observed pair  = MLP(left, right, left*right, abs(left-right), pair identity)
residual       = zero-init MLP(unary mean, active-pair mean, pattern identity)
graph node     = Original Slot node + residual
```

残差输出层为零初始化，因此相同共享参数下，训练开始时 Treatment 与 Control 输出完全相同。
Loss、MMoE、EMA Teacher、graph、mask、上游冻结特征和推理协议均不改变。

## 锁定协议

- 数据集：CMU-MOSI，fold 1；
- 特征：wav2vec-large-c、DeBERTa-large-4、MANet，全部冻结；
- 训练：all-rates-per-batch，一个 checkpoint 测试 missing rate `0.0,...,0.7`；
- seeds：66--70；100 epochs；batch size 32；
- Slot、Regression-MSE、DualGate MMoE、hidden 200、latent 256；
- window past/future 均为 2，time attention 关闭；
- Adam learning rate `5e-4`，由此前 validation LR 筛选锁定；
- checkpoint 只按 validation 八-rate W-F1 均值选择；
- Control 继承既有五种子 `5e-4` 结果，不重新训练。

正式通过条件同时要求：八-rate mean delta 为正、高缺失 `0.4--0.7` mean delta 为正、
两项都至少 3/5 seed 为正、miss0 mean decline 不低于 -0.3 个百分点。

## 五种子逐 missing-rate Test W-F1

| Miss | PCIR | Control | Delta (pp) | 正向 seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 85.708 | 85.616 | +0.092 | 3/5 |
| 0.1 | 83.662 | 83.544 | +0.118 | 2/5 |
| 0.2 | 81.534 | 81.085 | +0.450 | 3/5 |
| 0.3 | 80.129 | 80.216 | -0.087 | 2/5 |
| 0.4 | 77.320 | 77.393 | -0.074 | 2/5 |
| 0.5 | 75.163 | 75.310 | -0.147 | 1/5 |
| 0.6 | 74.806 | 74.907 | -0.101 | 2/5 |
| 0.7 | 72.620 | 72.873 | -0.254 | 2/5 |

逐 rate 的 PCIR sample standard deviation 为：

```text
0.676, 1.159, 1.148, 1.018, 1.984, 0.373, 1.651, 2.277
```

## 逐 seed 配对结果

| Seed | Best epoch | Val8 W-F1 | 八-rate Delta | High Delta | Miss0 Delta |
|---:|---:|---:|---:|---:|---:|
| 66 | 44 | 78.658 | +0.320 | +0.257 | +0.901 |
| 67 | 34 | 77.963 | -0.429 | -0.350 | -0.357 |
| 68 | 49 | 77.805 | +0.262 | +0.507 | -0.575 |
| 69 | 46 | 78.671 | +0.355 | +0.130 | +0.258 |
| 70 | 40 | 77.178 | -0.510 | -1.262 | +0.233 |

聚合结果：

| 聚合 | PCIR | Control | Delta | 正向 seeds | 门槛 |
|---|---:|---:|---:|---:|---|
| 八-rate mean | 78.8678 | 78.8680 | -0.0002 | 3/5 | FAIL |
| High `0.4--0.7` | 74.9769 | 75.1207 | -0.1438 | 3/5 | FAIL |
| Miss0 | 85.7085 | 85.6163 | +0.0921 | 3/5 | PASS |

PCIR 的总体 delta 几乎为零，但 seed 间 overall delta 的 sample standard deviation 为
0.430 个百分点；高缺失均值下降 0.144 个百分点。因此它不是稳定改进。

## 完整性审计

- 5/5 任务完成 100 epochs，best epoch 均来自 validation；
- 40/40 prediction NPZ 独立重算 W-F1，与 `metrics.json` 完全一致；
- 40/40 NPZ availability SHA 与本次 metrics 一致；
- 40/40 Treatment mask SHA 与相同 seed/rate 的继承 Control 一致；
- 每个 seed 均记录 `node_interaction_residual=true`；
- Treatment 参数量 32,292,645，Control 32,089,733，仅新增 202,912（0.632%）；
- V100 真实前后向有限，共享参数初始化最大绝对误差为 0；
- 远程相关回归测试：190 passed；warning 均来自 PyG/Matplotlib 上游弃用提示。

仓库级 `pytest tests` 另有一个与 PCIR 无关的既有收集错误：
`test_fixed_mask_bank.py` 导入当前源码中不存在的 `FixedConversationMaskSchedule`。PCIR 修改的
Missing-M3、text-LoRA、PLCI 相关 8 个测试文件已全部通过；本实验没有借机修改旧 mask-bank
接口，避免扩大单变量实验范围。

GPU0/2/3 在启动时已被其他 24.8 GiB 任务占用，所以正式运行使用 GPU1/5/6/7，
其中 GPU1 并行 seed66 与 seed70；故障 GPU4 未使用。该调度只影响运行时间，不改变配置。

## 结论

`COMPLETE — FAIL`。

PCIR 没有通过预注册的总体和高缺失门槛。现有证据只支持“显式 pair residual 在部分
seed/rate 改变结果”，不支持“它稳定改善 MOSI”。按照锁定规则，不为 PCIR 追加 attention、
gate、额外 loss 或其他救援模块，也不扩展到更多数据集。

结果位置：

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_pcir_node_20260829/formal`；
- Local：本实验目录的 `results/formal`；
- checkpoint 仅保留在 biggpu，不提交 Git。
