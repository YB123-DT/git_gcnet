# OSRAM 来源身份补全：相关分析索引

整理日期：2026-09-09。本次仅上传笔记并关联已有证据，不实现、不训练新方案。

## 方案原文：尚未验证

[保留观测/补全身份的三槽位融合](OSRAM_SOURCE_IDENTITY_COMPLETION_IDEA.md)

思路是让预测 latent 参与分类，但明确标注其来源为“补全”，不把来源身份
误当成置信度。第一遍 OSRAM 仍只读取真实可见输入，不增加第二遍 OSRAM。
融合结构、表示空间和尺度对齐尚未锁定，不能标为已实现或已提分。

## 已有实验与分析

| 记录 | 与本方案的关系 | 不能据此声称 |
|---|---|---|
| [旧 GCNet 分类回灌实验](../experiments/missing_m3_mosi_classification_completion_20260829/EXPERIMENT.md) | 训练、测试保留 Predictor，将预测 latent 投影为 hidden residual；八率均值77.581 vs78.059，负结果 | 它不是当前 OSRAM 的来源身份三槽位融合，不能替代新方案实测 |
| [OSRAM MMoE 4E/1E 对照](../experiments/mmoe_single_expert_mosi_20260907/RESULT.md) | 逐rate Test-oracle：80.781 vs80.851；单checkpoint八率均值约80.267 vs80.267 | 不能据小差值断言路由失效或回灌一定有效 |
| [MOSI miss=0.5 pattern 分析](../experiments/osram_mosi_pattern_20260909/RESULT.md) | eta.6重训练的下降集中在无Text的A/V/AV；V-only五seed均下降 | 它只定位退化子集，不证明缺失Text补全器能修复它 |
| [冻结写入步长曲线](../experiments/osram_write_step_range_20260909/FINDINGS.md) | 两数据集采样均值峰值在eta.6，但新旧association拟合误差均变大 | 不能把任务收益解释成补全更准确或历史association保存更好 |

上述结果的 missing schedule、主干和 checkpoint 选择规则并不全部相同。
它们是不同问题的历史证据，不应直接拼成同一张公平性能对比表。

## 继续讨论时需区分

- 来源身份：真实观测还是预测补全；不是“这个预测有多可信”。
- 当前正式推理路径：没有把 MMoE 预测 latent 回灌分类。
- 本笔记提议的路径：训练和测试均保留补全融合，Teacher仍只用于训练。
- 已有负面回灌结果只是风险参考；当前OSRAM版本的具体补全方案尚未实验。

本索引不新增训练任务、不更新主方法结论、不自动启动回灌实验。
