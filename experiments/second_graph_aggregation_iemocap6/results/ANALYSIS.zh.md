# 第二图层机制结果分析

主分析记录：[ANALYSIS.md](ANALYSIS.md)。英文版：[ANALYSIS.en.md](ANALYSIS.en.md)。

## 证据边界与实验设置

下文所有数字均由 [summary.json](summary.json) 重新计算，不从文件名推断，也不挑选最佳运行。协议与 provenance 边界记录在 [EXPERIMENT.md](../EXPERIMENT.md)，中文和英文镜像分别为 [EXPERIMENT.zh.md](../EXPERIMENT.zh.md) 与 [EXPERIMENT.en.md](../EXPERIMENT.en.md)。

比较固定为 IEMOCAPSix、第 5 fold、missing rate `0.0` 和 `0.7`、正式 seeds `66`、`67`、`68`。每个候选任务按照 rate、seed、fold 与 mask SHA256，和只读继承的 Original 档案配对。因此，每个 rate 有 3 次配对运行，每个候选共有 6 个配对差值。除非另有说明，数值均为 weighted F1 均值 ± **样本标准差**。Original 没有重新训练。

## 各 missing rate 的 weighted F1

| 实验臂 | Missing 0.0 | Missing 0.7 |
|---|---:|---:|
| Original | 0.627564 ± 0.004537 | 0.616087 ± 0.017411 |
| GenAgg | 0.476453 ± 0.235104 | 0.391534 ± 0.133442 |
| Scaled Soft Medoid | 0.626670 ± 0.017790 | 0.608371 ± 0.024686 |
| SSMA Conv2 | 0.623863 ± 0.021238 | 0.605441 ± 0.017788 |
| RTDR | 0.632265 ± 0.012190 | 0.616468 ± 0.034939 |

候选减去配对 Original 的差值为：

| 实验臂 | Missing 0.0 的差值 | Missing 0.7 的差值 | 跨 rate macro 差值 |
|---|---:|---:|---:|
| GenAgg | -0.151111 ± 0.237563 | -0.224553 ± 0.138094 | -0.187832 |
| Scaled Soft Medoid | -0.000893 ± 0.020555 | -0.007715 ± 0.031507 | -0.004304 |
| SSMA Conv2 | -0.003701 ± 0.021211 | -0.010645 ± 0.016442 | -0.007173 |
| RTDR | +0.004701 ± 0.007659 | +0.000381 ± 0.045653 | +0.002541 |

## Seed 稳定性

下表每一项是同一 seed 在 missing rate 0.0 与 0.7 上配对差值的均值。

| 实验臂 | Seed 66 | Seed 67 | Seed 68 | 正向 seeds |
|---|---:|---:|---:|---:|
| GenAgg | -0.036149 | -0.377294 | -0.150053 | 0/3 |
| Scaled Soft Medoid | +0.018302 | -0.022715 | -0.008500 | 1/3 |
| SSMA Conv2 | -0.001779 | -0.008314 | -0.011427 | 0/3 |
| RTDR | +0.029813 | -0.001006 | -0.021183 | 1/3 |

因此，RTDR 只有在把全部 6 个单元平均后才略微为正，其符号没有跨 seed 复现。尤其在高缺失率下，它的均值相对不确定性很小：均值差值 +0.000381，而样本标准差为 0.045653。

## 参数与运行时间

当前候选档案同时记录 `parameter_count`（当前实例化模型的参数数）和 `selected_path_parameter_count`（当前实验路径实际选中的参数数）。历史 Original 档案只有 selected-path 参数数具有可比语义；其 legacy `parameter_count` 字段不能解释成当前模型的总参数数，也不能放在“总参数数”列里与候选横向比较。每个候选的运行时间由 6 个任务汇总；Original 的 6 个唯一继承任务只统计一次。

| 实验臂 | 总参数数 | Selected-path 参数数 | 运行时间（秒） |
|---|---:|---:|---:|
| Original | N/A（legacy archive 只保存 selected path） | 34,140,166 | 362.439 ± 53.831 |
| GenAgg | 36,419,934 | 34,140,284 | 697.742 ± 160.192 |
| Scaled Soft Medoid | 36,419,816 | 34,140,166 | 447.302 ± 114.032 |
| SSMA Conv2 | 37,015,216 | 34,735,566 | 501.393 ± 109.075 |
| RTDR | 36,419,816 | 34,140,166 | 371.226 ± 145.975 |

这些 wall-clock 数值描述的是已完成任务，不是隔离的算子延迟；并发调度也可能增加波动。因此不能把它们当作严格受控的速度基准。

## 坍塌审计

按照预注册的类别覆盖判据，候选任务中恰好有一个被标记为坍塌：

| 实验臂 | Missing rate | Seed | 类别覆盖 | 主导预测比例 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| GenAgg | 0.0 | 67 | 4/6 | 0.498460 | 0.205430 |

Soft Medoid、SSMA 和 RTDR 均没有丢失类别覆盖。GenAgg 的失败受到这个坍塌单元（配对差值 -0.425189）以及高缺失率 seed 67、68 两个单元（差值 -0.329398 与 -0.276176）的明显驱动。但失败并非只由一次坍塌造成：GenAgg 的 6 个配对差值全部为负。

## 探索性配对检验

为完整披露，我们对每个候选的 6 个配对差值进行了双侧单样本配对差值 t 检验和双侧 Wilcoxon 符号秩检验。Bonferroni 校正在**各自的检验 family 内**进行，而不是把 8 项检验合并成一个 family：4 项 t 检验构成一个 family，4 项 Wilcoxon 检验构成另一个 family。因此，每个 family 的阈值均为 0.05 / 4 = 0.0125，相应校正 p 值均为 `min(4p, 1)`。

| 实验臂 | t(5) | 原始 t p | Bonferroni t p | Wilcoxon W | 原始 Wilcoxon p | Bonferroni Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| GenAgg | -2.5792 | 0.0495 | 0.1979 | 0 | 0.0313 | 0.1250 |
| Scaled Soft Medoid | -0.4378 | 0.6798 | 1.0000 | 8 | 0.6875 | 1.0000 |
| SSMA Conv2 | -1.0101 | 0.3588 | 1.0000 | 5 | 0.3125 | 1.0000 |
| RTDR | +0.2119 | 0.8405 | 1.0000 | 10 | 1.0000 | 1.0000 |

这些检验只是**探索性、低功效**分析（`n=6`）。更重要的是，6 个单元并不完全独立：同样的 3 个 seeds 在两个 missing rates 下重复出现，而且所有候选复用了相同的继承 Original 单元。因此，该表不能作为确认性推断。较小的原始 p 值不能证明机制有效，较大的 p 值也不能证明等价。没有任何比较越过 Bonferroni 校正阈值。

## 与数据一致的机制解释

- **GenAgg：** [GenAgg](https://arxiv.org/abs/2306.13826) 用可学习的广义聚合替换固定 sum。在本次适配中，这种额外自由度没有带来稳定的 GCNet 行为：所有配对单元都下降，一次完整模态运行只覆盖 4 个类别，而且高缺失率的两个最大跌幅在未触发类别覆盖坍塌时仍然存在。证据支持停止这个适配，但不能据此确定是哪个 GenAgg 内部参数造成不稳定。
- **Scaled Soft Medoid：** [鲁棒 soft-medoid 聚合](https://arxiv.org/abs/2010.15651)旨在限制孤立异常值的影响。这里的平均效应较小且为负，不同单元正负混合，只有一个 seed macro 为正。这说明在当前图与缺失协议下，对第二层消息做鲁棒中心化没有形成可复现优势；它并不能证明数据中不存在异常消息。
- **SSMA Conv2：** [SSMA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html)在压缩前增加跨邻居交互。三个 seed macro 都略微为负，因此更丰富的交互没有通过本轮 gate。由于 selected path 还增加了 595,400 个参数，这个负结果不支持继续做预注册的 parameter-matched follow-up；该控制只在正向 gate 后才需要。
- **RTDR：** RTDR 是自定义的对角 relation-transition 路由假设，不是 MrMP 迁移；检查过的 [MrMP 来源](https://arxiv.org/abs/2202.04844)会在层内混合关系。RTDR 的 macro 差值为 +0.002541，两个 rate 均值都为正，但只有 seed 66 跨 rate 提升。保守结论是出现了微小但 seed 不稳定的信号，而不是验证了改进。

## 预注册判定

锁定 gate 要求每个候选同时满足：两个 rate 的配对均值为正、跨 rate macro 差值为正、3 个 seed macro 至少 2 个为正、输出 finite 且无坍塌。GenAgg 未满足 rate、seed 和无坍塌条件；Soft Medoid 与 SSMA 未满足 rate 和 seed 条件；RTDR 未满足至少两个正向 seed 条件。因此，四个候选都在第一波停止。

其余 missing rates 不再运行。在看到预注册 gate 失败后扩展网格，会削弱停止规则、增加事后选择自由度，并继续消耗算力在没有达到最低稳定性要求的机制上。这是协议判定，不表示未经测试的 rates 必然为负。
