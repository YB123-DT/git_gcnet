# MOSEI Local / Base / Gap 诊断解读

INTERNAL DIAGNOSTIC ONLY

## 对象与口径

不是重新训练，也不是原 Flat 与新模型的架构比较。对象是已训练完的
**Source-Role Direct seed66**：60epochs、constant LR1e-3、原始
gcnet-random-mask-v1。原训练源码eb0e81e489a06a3f282dd5e57a1fb59f32ee8146；
本次只读诊断源码7840d8f388ac5d73eac9667f048a00a573961447。
各rate复用已保存的W-F1 Test-best权重及exact test mask，不重新选点。
同rate的四个读出使用相同Local、Memory轨迹、学习到的gamma和分类参数。
八率并非同一个checkpoint的部署评价。

L保留原Local adapter输入和原Local Skip；LB只增加Base；LG只增加当前
激活的Gap；Full同时保留Base与Gap。没有重新求Query或改写Memory。

## 观察

八率W-F1：L76.256、LB81.818、LG82.513、Full83.244。
高缺失W-F1：L68.900、LB77.750、LG79.699、Full79.980。

- Full对比L：八率+6.987pp，高缺失+11.080pp。
- 已有Base后再加Gap：八率+1.426pp，高缺失+2.231pp。
- 已有Gap后再加Base：八率+0.731pp，高缺失+0.281pp。
- 0.0时没有active Gap，L=LG、LB=Full，符合接口预期。
- 0.5和0.7时Full的ACC略低于LG（均低0.055pp），但W-F1仍略高；
  因此不能把W-F1变化当作每个rate的ACC也一致提高。

八率重复曝光计数中，LB→Full纠错789/致错479；LG→Full纠错614/致错436。
高缺失时LG→Full纠错165/致错152，边际净纠错较小；而LB→Full为491/322。
这些是同一测试话语在不同rate下的重复曝光，不是独立样本数量或显著性证据。

## 能说与不能说

在当前权重的最终分类输入干预下，去掉历史读值明显损害预测，尤其高缺失；
不能把本模型描述成“Memory完全没有作用”。Gap在已有Base之上仍有净增量；
Base在已有Gap之上的边际增量较小。二者并非两个可直接相加的独立收益。

去读值可能制造分布外输入。这不能证明分别重训Local-only一定只有76.256，
也不能据此定位写入、Query或融合哪个环节存在缺陷。该结果同样不证明
Source-Role校准优于原Flat：此前同条件原Flat83.279，本Direct83.244。
这里检验的是现有模型对历史证据的使用，不是新增机制的总体有效性。

## 执行与证据

biggpu健康GPU7，原s0环境，176次Memory扫描（8rate×22batch）。
用时114.174秒，Torch峰值分配显存469.704MiB；不代表整卡nvidia-smi占用。
22项针对性测试通过（含旧回归），1条第三方torch_geometric弃用警告。
八率Full均逐样本精确复现原预测；Frozen参数、RNG、原checkpoint与预测文件未改。

远程输出：
`/data2/yb/remote_experiments/osram_mosei_direct_role_readout_20261021/execution_v1`。
完整逐样本NPZ只留远程；Git保存RESULT、REPORT、METRICS、CHANGES及SUMMARY。
源快照9572个Git blob核对无差异。无新增训练、额外种子或自动调参。
