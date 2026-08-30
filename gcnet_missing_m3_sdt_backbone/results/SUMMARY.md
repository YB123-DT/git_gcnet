# 等活跃参数 SDT 主干正式结果

结论：**`CLOSED — NO IMPROVEMENT`**。在冻结 wav2vec、DeBERTa、MANet 特征及相同
Missing-M3、mask、loss、优化器和 mixed-rate 协议下，用等活跃参数的全上下文
Transformer 整体替换 GCNet 对话主干，没有通过任何一项主要性能门槛。

## 主结果

| 指标 | SDT candidate | Inherited GCNet control | 差值/门槛 | 判定 |
| --- | ---: | ---: | ---: | --- |
| Validation 8-rate mean W-F1 | 77.56 ± 1.03 | 78.77 ± 1.63 | −1.21；要求 ≥79.27 | FAIL |
| High-missing validation W-F1 | 73.53 | 74.96 | −1.43；要求不下降 | FAIL |
| Miss-0 validation W-F1 | 84.91 | 85.65 | −0.74；允许最低 −0.30 | FAIL |
| 正向 seed | 1/5 | — | 要求 ≥4/5 | FAIL |
| 非坍塌 | 通过 | — | 40 个 test 条件均输出两种符号 | PASS |

配对 validation delta 为 −1.21 ± 1.80 个百分点，95% CI 为 [−3.44, 1.02]；配对
t 检验 `p=0.2068`，Wilcoxon `p=0.1875`，Cohen's `dz=-0.673`。五个 seed 的统计功效
有限，但预注册门槛已明确失败，不需要依靠显著性检验作决定。

## 逐 seed

| Seed | Candidate best epoch | Candidate val8 | Control val8 | Delta | Candidate test8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 17 | 76.61 | 77.31 | −0.70 | 77.34 |
| 67 | 25 | 78.52 | 78.68 | −0.16 | 79.06 |
| 68 | 19 | 78.72 | 78.22 | +0.50 | 80.23 |
| 69 | 30 | 77.41 | 81.54 | −4.13 | 78.39 |
| 70 | 32 | 76.53 | 78.09 | −1.55 | 78.48 |

## 五种子逐 missing rate 均值

| Missing rate | Candidate validation | Control validation | Delta | Candidate test |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 84.91 | 85.65 | −0.74 | 85.96 |
| 0.1 | 82.84 | 84.53 | −1.68 | 83.87 |
| 0.2 | 80.30 | 81.71 | −1.41 | 81.23 |
| 0.3 | 78.31 | 78.42 | −0.12 | 79.59 |
| 0.4 | 77.02 | 77.14 | −0.13 | 76.95 |
| 0.5 | 72.76 | 74.83 | −2.07 | 74.86 |
| 0.6 | 72.66 | 74.06 | −1.40 | 73.70 |
| 0.7 | 71.67 | 73.81 | −2.14 | 73.43 |

Candidate 在八个 validation rates 的聚合均值上全部低于 control；差距在 `0.5` 和
`0.7` 最大。因此不能用“只在完整条件下降、缺失条件提升”解释本结果。

## 诊断含义

1. 单纯把 GCNet 换成全上下文 Transformer 不是 MOSI 提分答案；这条路线正式关闭。
2. Candidate 的最佳 epoch 为 17–32，明显早于 control 的 42–50，说明 attention 主干
   更快拟合，但没有形成更好的跨 seed 泛化。
3. 所有 loss 有限，test prediction std 为 0.774–1.476，所有 40 个 seed-rate 条件都
   预测出两种符号，因此失败不是数值坍塌或常量输出造成的。
4. 下一步不应继续更换第三个 conversation backbone；证据更支持检查冻结模态特征进入
   下游表示的方式，以及 text-dominant MOSI 中跨模态辅助目标是否稀释情感判别信号。

## 审计限制

Candidate 的 8 个 test mask SHA 已保存并通过 manifest 核对；但 inherited control 是
validation-only 运行，其 `history.json` 没有保存 validation mask SHA。因此 validation A/B
的逐 mask 配对只能由相同 seed、相同配置和双方共享的确定性 `_schedules()` 实现推断，不能
仅凭现有 artifact 做密码学复核。该限制不改变本次门槛失败，但必须在论文或总账中披露。

Test 仅在预先由 validation 选择的 checkpoint 上报告，没有用于反向修改模型。完整机器可读
结果见 [`SUMMARY.json`](./SUMMARY.json)，运行证据见 [`formal/manifest.json`](./formal/manifest.json)
与 [`PROVENANCE.json`](./PROVENANCE.json)。
