# IEMOCAP-6 stage-aware mask 协议恢复与正式比较

日期：2026-08-24

## 问题与根因

旧的 fixed-bank 实现把同一份 utterance mask 同时用于 100 个训练 epoch、
validation 和 test。它保持了方法间配对，却移除了 GCNet 原训练路径在固定缺失率
下不断变化具体缺失组合的训练增强。其 Original 在 missing=0.5/0.7 的五 seed
均值降至 0.5782/0.5512。

修正协议保持 requested missing rate 不变，并使用：

- 100 份预生成、跨方法共享的逐 epoch 训练 mask；
- 1 份独立固定 validation mask；
- 1 份独立固定 test mask；
- rate、seed、fold 相同的 Original、Full、CP-LECC 使用相同 bundle hash。

## Original 恢复门

IEMOCAPSix、fold 5、100 epochs、seeds 66--70，biggpu
`gcnet-official`（Python 3.8.20、Torch 1.8.0、PyG 2.0.1）。

| Missing | 旧复现同 seeds | stage-aware Original | 差值 | 恢复门 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.608924 | 0.613622 | +0.004698 | PASS |
| 0.7 | 0.611208 | 0.603929 | -0.007279 | PASS |

预注册容差为每档不低于旧同-seed均值 0.015。10/10 任务成功，所有任务预测类别
覆盖为 6。根因因此得到判别性支持：此前低基线主要来自单一训练 mask 的重复使用，
而不是数据、fold 或 CP-LECC 代码。

## 修正协议正式结果

| 方法 | Missing 0.5 W-F1 | Missing 0.7 W-F1 | 两档均值 |
|---|---:|---:|---:|
| Original | 0.613622 ± 0.015644 | 0.603929 ± 0.025612 | 0.608776 |
| Full FiLM | **0.623174 ± 0.011146** | 0.603523 ± 0.021231 | **0.613349** |
| CP-LECC | 0.610105 ± 0.014600 | **0.613778 ± 0.009217** | 0.611941 |

CP-LECC 相对 Original：

- missing=0.5：-0.003517；
- missing=0.7：+0.009849；
- 两档总体：+0.003166；
- 逐 seed 两档平均为正：2/5。

CP-LECC 相对 Full：missing=0.5 为 -0.013069，missing=0.7 为 +0.010255。

30/30 任务成功、全部 coverage=6，三臂逐 rate/seed 的 bundle hash 完全一致。

## 判定

CP-LECC 未通过锁定门槛：0.5 下降、总体提升不足 0.005，且只有 2/5 seeds
为正。它只能支持“edge-conditioned correction 对极高缺失率有帮助”这一局部结论，
不能作为最终统一方法。

Full FiLM 也不能直接成为最终方法：它在 0.5 提升 0.009552，但在 0.7 下降
0.000406，总体提升 0.004573，仍低于 0.005 门槛。

五 seed 配对检验中，Full vs Original 在 0.5 的 paired t-test 为
`p=0.0486`，Wilcoxon 为 `p=0.0625`；其余逐 rate 比较均未达到 0.05。由于样本量
只有 5，不把边界 p 值解释为稳定显著性。

## 七种 pattern 分层

按保存的固定 test mask 分层后，CP-LECC 在 0.7 的收益主要来自 V-only：

| Missing | Pattern | Full − Original W-F1 | CP-LECC − Original W-F1 |
|---:|:---:|---:|---:|
| 0.5 | A | +0.0096 | -0.0147 |
| 0.5 | T | +0.0123 | +0.0120 |
| 0.5 | V | +0.0128 | -0.0100 |
| 0.5 | AT | +0.0088 | +0.0153 |
| 0.5 | AV | +0.0289 | -0.0059 |
| 0.5 | TV | +0.0006 | -0.0006 |
| 0.5 | ATV | -0.0262 | -0.0348 |
| 0.7 | A | -0.0018 | -0.0034 |
| 0.7 | T | +0.0021 | +0.0059 |
| 0.7 | V | -0.0014 | +0.0275 |

这说明 CP-LECC 不是统一改善所有缺失模式：它在纯单模态邻域中的 V-only/T-only
传播有价值，但在 0.5 的混合 pattern 邻域会伤害 A-only、V-only 和 ATV 接收节点。
Full 的优势则集中在 0.5 的单模态及 AV 模式。后续机制必须显式处理 pattern
regime，而不是继续增大统一边修正器。

此前 single-fixed-bank 结果保留为协议诊断，不进入最终模型证据。

## 证据目录

- 修正协议结果：`/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/protocol_recovery_v1_biggpu`
- single-fixed-bank 诊断：`/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/formal_v1_biggpu`
- 旧 IEMOCAP-6 复现：`/data2/yb/paper/GCNet_repro_iemocap4_fold5_10seed_20260819/experiments/gcnet_all_missing_10trials_20260819`
