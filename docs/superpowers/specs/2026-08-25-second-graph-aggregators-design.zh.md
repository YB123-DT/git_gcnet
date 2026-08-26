# GCNet 第二层聚合器设计（中文）

主设计：[2026-08-25-second-graph-aggregators-design.md](2026-08-25-second-graph-aggregators-design.md)。英文镜像：[2026-08-25-second-graph-aggregators-design.en.md](2026-08-25-second-graph-aggregators-design.en.md)。

## 锁定结论

本轮只替换 GCNet temporal 与 speaker 两个分支的第二层 `GraphConv` 邻域聚合器：GenAgg 为主候选，带尺度修正的 Soft Medoid 为备选。第一层 RGCN、图拓扑、relation、BiLSTM、分支加法、重建、分类器、优化器和自然缺失协议全部不变。显式self edge继续作为邻居参与聚合，同时保留独立root路径；不新增、删除或去重self edge。已有 Original 40 个 NPZ 直接继承，不重新训练。

## 文献与撞车检查

- Kortvelesy、Morad、Prorok，GenAgg，NeurIPS 2023：[论文](https://arxiv.org/abs/2306.13826)，[论文时期官方代码](https://github.com/Acciorocketships/generalised-aggregation/blob/3c95c10afac4bda77afc30e80a7481c7e537fca1/genagg/genagg.py)。
- Geisler、Zügner、Günnemann，Soft Medoid，NeurIPS 2020：[论文](https://arxiv.org/abs/2010.15651)，[官方代码](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/blob/4f94140afb7fd2ef5bf77f45a5efc7b2d6eb2a09/rgnn/means.py)。

2026-08-25 使用 GenAgg、generalised f-mean、Soft Medoid、medoid、learnable aggregation、robust aggregation 与 multimodal sentiment、emotion recognition、emotion 的组合检索 arXiv，精确结果均为零；本地 MSA/MERC/ERC 文献语料也没有精确机制匹配。这只能支持“跨领域迁移”，不能写成绝对首次使用。

## GCNet 真实插入位置

官方 biggpu 环境中的 PyG 2.0.1 已确认：第二层 `GraphConv` 默认使用 `add`，即

\[
y_i=W_n\sum_jx_j+W_rx_i+b.
\]

新方法只替换邻居聚合，保留 neighbor transform、root transform 和 bias。两个图分支分别拥有自己的聚合器。锁定窗口为 2/2，一个内部节点最多五条入边，包括显式 self edge。

## GenAgg

\[
\operatorname{GenAgg}(X_i)=
f^{-1}\left(n_i^{\alpha-1}\sum_jf(x_j-\beta\mu_i)\right).
\]

严格采用论文实验的 `1-2-2-4` 正向 MLP、`4-2-2-1` 逆向 MLP、Mish、BatchNorm、仅对Linear weight执行Kaiming初始化、保留Linear bias默认初始化及可学习 \(\alpha,\beta\)。令 \(x_c=x_j-\beta\mu_i\)，精确inverse目标为 \(\operatorname{mean}[(|f^{-1}(f(x_c))|-|x_c|)^2]\)。兼容实现复用聚合时已计算的 \(f(x_c)\)，不在hook里二次运行encoder，因此BatchNorm统计只更新一次；这是相对发布hook的明确工程偏差，但保持论文损失与梯度。temporal与speaker两项loss求和后仅一次以权重1.0并入训练总损失，任何forward内都不调用 `.backward()`。

官方训练环境 Torch 1.8/PyG 2.0.1 没有新版 Aggregation API 和 `nn.Mish`，因此使用 `torch_scatter` 和 `x*tanh(softplus(x))` 做数学等价兼容实现，不新增依赖。论文实验的BatchNorm配置每分支新增59个参数（正向Linear 22、逆向Linear 19、BN affine 16、两个标量），两个分支共118个；当前包默认关闭BN，不是本实验选择。新增参数初始化必须使用独立 RNG 分叉，保证所有 GCNet 共享参数与 Original 完全同初始化。

理由：缺失会改变第一层关系消息的邻域分布。GenAgg 可以学习基数依赖、中心化与非线性统计，不要求完整邻居占多数，因此比 Soft Medoid 更适合作为主候选。

## Soft Medoid

\[
d_j=\sum_k\lVert m_j-m_k\rVert_2,
\quad s_j=\operatorname{softmax}(-d_j/T),
\quad \operatorname{Agg}=n_i\sum_js_jm_j.
\]

初次实验固定来源默认 `T=1.0`。乘以真实入边数 \(n_i\) 后，高温极限才恢复 Original sum；单邻居和同质邻域也严格等于 add。neighbor linear 在距离计算前无 bias 地执行，bias 聚合后只加一次，root 路径不变。打包padding既不进入距离和，也不进入候选softmax；零度节点跳过softmax，邻居项为零，输出严格为 `lin_r(x_i)+lin_l.bias`。实现只构造 `[N,max_degree,D]`，不建立 `[N,N,D]`，不使用 top-k，相对GraphConv新增参数为零。

理由：缺失消息可能成为几何离群点。风险是 GCNet 邻域只有3至5个点，且缺失率0.7时不完整消息通常不是少数，因此它只作为备选。

## 不采用的方案

- 不把 GenAgg 改为 sum-preserving 初始化，因为这会偏离首次来源机制验证。
- 不加 mask-conditioned GenAgg/Medoid，因为会与 MPFiLM、CP-LECC 的缺失门控混杂。
- 首轮不做 temperature grid、parameter-match、edge attention、动态图或无关额外损失。

## 最小验证

不跑任何 1-epoch smoke，只做：Original add 的参数/RNG/前后向等价；GenAgg 在eval/fixed-map下手算、`alpha=1,beta=0` sum特例、精确inverse-MSE梯度及参数数；Soft Medoid 手算、packed/ragged与排列等价、零度、单邻居/同质邻域等价及相对GraphConv零参数增量；双分支插入检查；Python 3.8 检查；同步到 biggpu 后一次 FP32 GPU 前后向。

唯一完整测试入口已经确认：

```bash
PYTHONPATH=gcnet:. /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest discover -s tests -v
```

biggpu 正式训练使用 `/data2/yb/reproduction_envs/gcnet-official/bin/python`。测试环境与训练环境职责不同，不再互换尝试。

## 判别与正式实验

锁定 IEMOCAPSix fold 5、三种既有特征、hidden 200、graph hidden 100、窗口2/2、100 epochs、seeds 66–70及固定 stage-aware mask bank。

第一波只新增12个任务：两个候选 × missing `{0.0,0.7}` × seeds `{66,67,68}`，四张卡每卡三个。必须使用 `stage=formal` 加显式缩小网格启动，因为现有runner会把stage写入路径和不可变command provenance；若使用 `stage=gate`，后续formal无法原位继承。Original 使用已有40个 NPZ，候选晋级后直接跳过这12个已完成formal路径，不重复训练。

历史 Original 早于当前 branch fusion、pre/post context、selected-path-count 与 second aggregation 字段。专用继承验证器把缺失历史字段解释为锁定默认值 `addition/bilstm/bilstm/add`，但仍严格验证原始run manifest、命令、fold、特征、seed、rate、参数数与mask SHA；候选继续使用当前严格payload验证。Original根目录必须与候选目录分离。

上述两档 rate × 三个 seeds 的规则只保留为历史初筛 gate。用户随后把每个已训练模块的最低证据统一提高为 missing `{0.0,0.5,0.7}` × seeds `{66,67,68,69,70}`，即每个候选15个 mask 配对单元。统一描述性判据要求：总体宏差值为正、三个 rate 的配对均值全部为正、五个 seed macro 至少三个为正、输出 finite 且没有坍塌；它不追溯改写历史 gate。通过这一共同证据底线的候选再补齐8 rates ×5 seeds，已完成单元继续继承而不重跑。

## 本地与远端同步

本地权威分支：

```text
/data2/yb/paper/GCNet_TPAMI/.worktrees/second-graph-aggregators
```

biggpu 执行副本：

```text
/data2/yb/paper/GCNet_second_graph_aggregators_20260825
```

两边 `/data2` 并不共享。训练前本地同步到 biggpu；结束后结果和 manifest 同步回本地；启动前与回传后比较源代码 SHA256 manifest。
