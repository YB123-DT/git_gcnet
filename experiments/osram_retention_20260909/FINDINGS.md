# 真实 checkpoint 诊断结果

完成 40 次 evaluation-only：IEMOCAP-4 / MOSI × seeds66–70 × rates0.1/0.3/0.5/0.7。
没有训练、调参或修改 memory 数学公式。共 314,200 条原始记录。

## 首份真实 JSONL

IEMOCAP-4 seed66 rate0.7，实际 checkpoint epoch33，selection 为既有
8-rate-mean Test-oracle；不是为本次诊断重新选点。
记录16,309条，连续缺失配对9,248个；三项 post/pre 指标最大差为0。
见 raw/iemocap4_seed66_rate0.7/retention.jsonl.gz 与 metadata.json。

全部40次的首batch logits 开关前后逐元素相等。总计134,904个连续缺失
head/query配对：error/cosine/norm_ratio 的 post(t) 与 pre(t+1) 最大差为0。
这是标量指标连续性检查，不声称用三项标量证明了任意向量相等。

## Paired difference 优先结果

每条记录定义 D=write_damage-decay_damage，保持 signed。
下表先计算每个seed/modality组内均值，再等权平均15个组；不是F1百分点。

| Dataset | Rate | Mean decay_damage | Mean write_damage | Mean D |
|---|---:|---:|---:|---:|
| IEMOCAP-4 | 0.1 | 0.017147 | 0.270916 | 0.253769 |
| IEMOCAP-4 | 0.3 | 0.014214 | 0.217227 | 0.203012 |
| IEMOCAP-4 | 0.5 | 0.011769 | 0.171830 | 0.160061 |
| IEMOCAP-4 | 0.7 | 0.009830 | 0.140769 | 0.130939 |
| MOSI | 0.1 | 0.019134 | 0.130997 | 0.111863 |
| MOSI | 0.3 | 0.016810 | 0.106803 | 0.089993 |
| MOSI | 0.5 | 0.015103 | 0.088734 | 0.073631 |
| MOSI | 0.7 | 0.013659 | 0.078966 | 0.065308 |

120/120个dataset/seed/rate/modality组的D均值为正。
逐conversation先平均再等权汇总，120/120组也均为正；这不是说每条记录
或每个conversation都为正。CSV保留中位数、正值比例和conversation计数。

## 可以说与不能说

当前探针定义下，最近一次真实observed association在后续missing步骤中的
单步相对误差增加，write项平均大于decay项，且跨种子、rate、数据集同方向。

不能推出：block write是F1下降的根因；decay无害；降低write一定提分。
没有分析旧association的任务相关性，也没有累计分解整个缺失区间；
err_post只影响后续读取，不是当前classification read。统计中的heads和
time points相关，未将它们当独立样本计算显著性。
地址overlap分桶表已导出但本轮不据此解释因果。

下一步留待用户决定；本次不自动修改任何模型机制。
