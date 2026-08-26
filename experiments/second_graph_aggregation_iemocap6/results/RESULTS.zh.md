# 第二层图机制配对实验结果

## 初始预注册判别门槛

PASS 表示满足全部初始晋级条件；FAIL 表示至少有一项条件未满足。下表保留当时的历史判定，不因后续补充实验而改写。

| 候选模块 | 是否晋级 | seed 宏平均 F1 差值 |
|---|---:|---:|
| genagg | FAIL | -0.187831724 |
| soft_medoid | FAIL | -0.004304363 |
| ssma | FAIL | -0.007173215 |
| rtdr | FAIL | +0.002541466 |

## Gate 后 RTDR 稳定性审计

用户要求的后续实验不改变初始 FAIL。15 对扩展满足较窄的 post-gate extension criterion（`+0.008510981`，3/5 个 seed macro 为正）。补齐 8 个 missing rate x 5 seeds 后，RTDR 的总体配对 macro 差值为 `-0.002810103`，仅 3/8 个 rate 均值为正、3/5 个 seed macro 为正。所有运行均 finite 且没有类别坍塌，但预定义字段为 `stable_positive=false`。

| 审计范围 | 配对单元 | 总体差值 | 正向 rates | 正向 seeds | 状态 |
|---|---:|---:|---:|---:|---|
| 扩展（rates 0.0、0.5、0.7） | 15 | +0.008510981 | 3/3 | 3/5 | extension criterion PASS |
| 完整（rates 0.0-0.7） | 40 | -0.002810103 | 3/8 | 3/5 | `stable_positive=false` |

逐任务证据见[扩展结果](rtdr_extension/RESULTS.zh.md)和[完整网格结果](rtdr_full/RESULTS.zh.md)。

## 统一三档缺失率、五种子对比

为使所有已实现模块都具有同样的最低证据网格，最终对比固定 missing rates `{0.0,0.5,0.7}` 和 seeds `{66,67,68,69,70}`，每个实验臂 15 个配对单元。`uniform_stable` 要求总体 macro 差值为正、3/3 个 rate 均值为正、至少 3/5 个 seed macro 为正，且全部运行 finite、无坍塌。这是单独的描述层，不改写初始 gate，也不改写 RTDR full-rate `stable_positive=false` 的结果。

| 候选模块 | 配对单元 | 总体差值 | 正向 rates | 正向 seeds | 无坍塌 | `uniform_stable` |
|---|---:|---:|---:|---:|---:|---:|
| genagg | 15 | -0.204847963 | 0/3 | 0/5 | 否 | `false` |
| soft_medoid | 15 | +0.004706753 | 2/3 | 4/5 | 是 | `false` |
| ssma | 15 | -0.001153174 | 1/3 | 2/5 | 是 | `false` |
| rtdr | 15 | +0.008510981 | 3/3 | 3/5 | 是 | `true` |

Soft Medoid 的总体差值小幅为正，但 missing `0.7` 均值差为 `-0.002089281`，因此未满足全 rate 为正的条件。RTDR 是唯一满足这一有限三档描述条件的模块，但其更广的 40 对审计仍是总体负值。逐任务证据见[统一三档结果](uniform_three_rate/RESULTS.zh.md)和 [summary.json](uniform_three_rate/summary.json)。
