# Missing-streak 累计分解：只分析既有记录

未调用模型、未训练、未修改任何 memory/fusion/loss。本报告来自原40个run的
314,200条JSONL记录。代码：analyze_missing_streaks.py。

## 区间定义和质量检查

以 (run, conversation, modality, last_observed_time) 定义gap，
last_observed_time=time_index-history_distance。按head分别重建，要求距离
严格为1…L，拒绝重复、断点和head不齐全；之后对head求均值得到gap级记录。

共22,098个gap，176,784个head-gap。排除2,512个NO_HISTORY query；
这些没有真实历史anchor，不能伪造为零损伤。当前observed时不记录旧probe，
因此结束原因标为 observed_or_end_unresolved，不能从JSONL准确区分二者。

err_start=首个missing步骤的err_pre；err_end=最后missing步骤的err_post。
明确最后一步post尚未用于该句分类。检查：

sum(decay_damage)+sum(write_damage)=err_end-err_start

最大闭合误差2.0559e-7，连续步骤的err_post/err_pre通过检查。
保留signed值，不把改善截成0。现有overlap只能汇总max-key-cosine，
不称为projection-overlap。逐gap保留overlap均值、最大值、和。

## 趋势：以run为汇总单位，不把head当独立试验

| Dataset | 长度 vs 累计write正相关run | 平均Spearman | 长度 vs 每步write正相关run |
|---|---:|---:|---:|
| IEMOCAP-4 | 20/20 | 0.4179 | 0/20 |
| MOSI | 19/20 | 0.2287 | 0/20 |

MOSI seed70 rate0.1是例外：Spearman=-0.04886，不能写成所有run成立。
先在conversation内算相关再等权汇总，方向仍为IEMOCAP20/20、MOSI19/20。
每步write的平均Spearman：IEMOCAP=-0.4077；MOSI=-0.2109。

## 长度分桶

下表均值为现有 dataset/seed/rate/modality/bucket 组的等权平均；
不同bucket可能缺少某些组，不能把表当严格配对的因果趋势。

| Dataset | Gap长度 | Gap数 | 累计write | 累计decay | 每步write |
|---|---|---:|---:|---:|---:|
| IEMOCAP-4 | 1 | 8562 | 0.25848 | 0.01915 | 0.25848 |
| IEMOCAP-4 | 2–3 | 4522 | 0.37193 | 0.02198 | 0.16910 |
| IEMOCAP-4 | 4–7 | 1149 | 0.49142 | 0.02343 | 0.10924 |
| IEMOCAP-4 | 8+ | 102 | 0.60306 | 0.02707 | 0.06877 |
| MOSI | 1 | 4612 | 0.13729 | 0.02061 | 0.13729 |
| MOSI | 2–3 | 2499 | 0.17876 | 0.03061 | 0.08065 |
| MOSI | 4–7 | 603 | 0.22187 | 0.04766 | 0.04959 |
| MOSI | 8+ | 49 | 0.31683 | 0.07437 | 0.03681 |

## 解释边界

长gap累计损伤更大有较稳定描述性证据，但求和项更多本身就会产生趋势。
不能声称长gap每一步干扰更强；平均每步损伤恰好降低，可能有饱和/边界效应，
本轮未验证其原因。8+样本很少；未对相关heads/时刻做虚假的独立显著性检验。
该累计分解是实际轨迹上的望远镜求和，不是分别禁用decay/write的反事实。

## 文件

- streak_heads.csv.gz：逐head完整累计分解与闭合误差（git中无损压缩）。
- streak_gaps.csv：head均值后的独立gap条目。
- streak_buckets.csv：按run/modality/长度桶的均值与中位数。
- streak_trends.csv：每run Pearson/Spearman及conversation内汇总。
- overlap_trends*.csv/md：独立地址重叠分析，见对应文件。

Protection 仍是待验证干预候选，未实现；后续若获批准，必须与global-write
weakening对照。即使overlap相关也不能直接证明保护旧地址会改善情感识别。

## 合并 overlap 证据后的判断

地址重叠与signed write_damage的Spearman在40/40个run为正，中位数0.55431。
去除modality×head×distance-bucket的联合组均值后，adjusted-rank相关
40/40为正，中位数0.57304。conversation内相关等权平均也40/40为正。
这比未分层相关更有支持力，但不是每个小分组都为正，也不是所有桶严格单调。
跨conversation均值相关只有35/40为正；详细负例和稀疏计数见overlap报告。

结论：有理由把保护缺失历史地址作为下一次可证伪的干预候选；不声称
mechanism已被因果证明。必须同时保留全局write weakening对照；观察
retention是否改善以及是否牺牲新信息学习，再看独立情绪预测结果。
本轮不实现任何干预，也不据此决定永久停止研究decay。
