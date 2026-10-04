# M01–M40 代码交付

INTERNAL DIAGNOSTIC ONLY — 本次仅代码，不是训练结果。

## 当前范围

用户指定的40项全部实现；另有22项此前完成的来源核对型实现。
这表示62个实现/配置入口，**不是62个新的、不重复的方法**，更不是已完成120项。
120仍是候选扩展目标；本次优先交付M01–M40，不以历史同机制或宽深变化凑数。

最新指令为先写代码、暂不启动新实验。已停止GPU2的PointCNN容量等待器，
已有训练保留，没有启动本批任何训练或GPU smoke。

## 接入位置

|类型|编号|实际行为|
|---|---|---|
|R|其余37项|五来源投影64维，算子输出64维，经零初始化64→1600桥，加到原Flat pre-norm anchor|
|I|M11、M13|仅改变Flat Adapter输入；原Local Skip继续使用原Local|
|N|M30|只把`emotion_adapter[0]`换成DyT；保留最终`emotion_norm`，不增加残差参数|

R/I新增初始化在现有`fork_rng`中隔离。R的零桥和I的恒等调制通过完整模型输出/RNG检查。
M30是非等价LN算子对照，不声称它在非零Adapter下与原Flat函数一致。
Memory读写、Query、任务头、loss、random-missing协议没有改变。
只用forward 512维做新增关系计算，固定Gap槽位，inactive/padding安全清零，首句跳过新增历史分支。

## 代码导航

- `gcnet_missing_m3/priority40_registry.py`：40个稳定ID与分组。
- `priority40_relations.py`：M01–M05、M36–M40。
- `priority40_bilinear.py`：M06–M10。
- `priority40_conditioning.py`：M11–M15。
- `priority40_pooling_geometry.py`：M16–M20、M26–M29。
- `priority40_crosses.py`：M21–M25。
- `priority40_mixers.py`：M31–M35。
- `priority40_common.py`：公共五来源投影与M30。
- `gcnet_missing_m3/osram.py`：复用已有meaningful开关，仅M30有专门替换分支。
- `configs.py`：从真实cfg84配置导出40份配置及逐项参数量，不启动进程。

## 配置

```bash
python -m experiments.osram_priority40_20261004.configs \
  --reference experiments/osram_priority40_20261004/BASELINE_CONFIG.json \
  --output /an/unused/output/directory
```

生成目录必须不存在。仅改变`osram_meaningful_block`，不重设seed、epoch、lr或batch。
训练入口已接受例如`--osram-meaningful-block m01_sab`；**本次不执行训练命令**。
`BASELINE_CONFIG.json`来自biggpu原Flat seed66，SHA256为
`65ba11e17dff20264d05f60932c00ae389f6efa7c3f0acf10720db659464a2ba`，与历史审计一致。
其checkpoint协议是per-rate Test-oracle内部筛选，不是正式validation选模。

## 验证与边界

CPU验证包含全部核心、固定mask、inactive NaN隔离、padding、有限梯度、更新、
39项初始化整模等价/RNG、M30替换范围及原实现回归。详见`VERIFICATION.json`。
参数量及配置文件见`configs/SUMMARY.json`。

全部是针对当前任务的机制移植，不宣称复现原论文整套网络。
源码文件保留原文/代码出处和明确适配差异。ASP、Cross-stitch未确认原作者代码；
Dynamic ReLU对照后续作者实现；TFN对照参考实现，均没有冒充原作者版本。
NODE只在首次有效训练输入初始化，eval不读取测试统计进行初始化。
历史同机制对应在配置摘要标记为对照，不计为新方法；未穷尽审查者标pending。

本批没有W-F1、新checkpoint或GPU稳定性结论；不以CPU通过推断性能提升。
