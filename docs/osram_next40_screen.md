# 新一批40条线索的去重筛查（不是40项合格实验）

LITERATURE SCREEN ONLY — NO NEW TRAINING

## 结果先说清楚

2026-10-04 新增检查40条论文线索：**9条暂留设计审查、17条待定或仅存档、14条排除**。这没有完成“40个有意义且方向不撞的合格方法”目标。暂留也不等于已经证明独立、适配、可训练或提分；**本次批准实施/训练数量为0**。现有训练和最多60个不同方法的上限均未改变。

对照基准为 [既有40项](osram40_inventory.md)、[机制注册表](../experiments/osram_method_registry.json)、[被用户撤回的基础20项](../experiments/osram_readout20_20261003/CANDIDATES.zh.md) 与 [旧重筛记录](../experiments/osram_readout20_reconsidered_20261003/RECONSIDERATION.zh.md)，并排除早期Gate/GRN/简单Relation、补全和双视图路线。只查题名不算机制去重，曾经否决的方法也不能换包装重新计数。

## 当前边界

只处理已有Local256、Base/Gap真正forward512及availability。保持OSRAM读写/query、原任务头和loss；不新增历史缓存、跨batch支持集、缺失预测或训练目标。不将head或channel当词序、时间或像素网格。改插入位置、深宽、rank、solver实现、基础算子外加MLP均不能独立占名额。

下面的“暂留”仅表示主方法和某份代码链有可核查依据，值得讨论具体设计。它不保证是作者原实验的同版本代码：DPPy是独立参考库，TensorFlow Lattice是作者项目后续实现；卡片逐项说明。部分原文/代码仍不可获得或不一致，明确pending。来源检查不是运行正确性检查；没有安装或执行这些仓库。

## 40条逐项筛查

|#|论文线索|当前判断|原因|
|---:|---|---|---|
|1|[约束QP / OptNet](https://proceedings.mlr.press/v70/amos17a.html)|待定／不计数|待定：若没有任务支持的耦合约束，只是旧隐式优化换求解器。|
|2|[DPP联合子集](https://arxiv.org/abs/1207.6083)|暂留设计审查|暂留：行列式建模整组互补性；不能缩水成独立marginal gate。|
|3|[SATNet](https://proceedings.mlr.press/v97/wang19e.html)|待定／不计数|待定：布尔命题与辅助变量的语义尚未成立，不能变成补全。|
|4|[几何中位数](https://doi.org/10.1109/TSP.2022.3153135)|待定／不计数|待定：与EA优化聚合方向重叠；不能只换稳健势。|
|5|[次模覆盖选集](https://arxiv.org/abs/1803.01785)|待定／不计数|待定：原方法代码、无集合监督下的梯度路径待核实。|
|6|[图总变差 / Lovasz层](https://papers.neurips.cc/paper_files/paper/2017/hash/192fc044e74dffea144f9ac5dc9f3395-Abstract.html)|待定／不计数|待定：作者代码为网格TV，当前语义图及相应反传没有闭合。|
|7|[DSAC](https://arxiv.org/abs/1611.05705)|排除|排除：几何假设求解与期望风险不能在当前接口和loss下忠实保留。|
|8|[SparseMAP](https://proceedings.mlr.press/v80/niculae18a.html)|待定／不计数|待定：尚无独立的合法结构空间，可能只是QP/OT替换solver。|
|9|[可微动态规划](https://proceedings.mlr.press/v80/mensch18a.html)|排除|排除：没有真实有序路径，不把heads假当时间。|
|10|[GroupDRO](https://arxiv.org/abs/1911.08731)|排除|排除：改训练风险与跨batch状态，不是局部读出模块。|
|11|[双曲几何网络](https://arxiv.org/abs/1805.09112)|暂留设计审查|暂留：基点相关Möbius运算；层级假设未被验证。|
|12|[PersLay持久同调](https://proceedings.mlr.press/v108/carriere20a.html)|待定／不计数|待定：从特征到diagram的可微链未闭合，不能只搬点集pool。|
|13|[BernNet谱响应](https://arxiv.org/abs/2106.10994)|暂留设计审查|暂留但近重复风险高：与旧图扩散及本批scattering同属谱/图方向。|
|14|[Diffusion Scattering](https://arxiv.org/abs/1806.08829)|暂留设计审查|暂留：固定多尺度wavelet、多阶模值路径，区别单次线性滤波。|
|15|[HRR绑定/解绑](https://arxiv.org/abs/2109.02157)|待定／不计数|待定：固定key整链可能化成线性层；动态key又接近简单双线性。|
|16|[LBDN Sandwich](https://proceedings.mlr.press/v202/wang23v.html)|暂留设计审查|暂留：耦合参数化约束分支敏感性，不是逐矩阵裁剪。|
|17|[Grassmann网络](https://doi.org/10.1609/aaai.v32i1.11725)|待定／不计数|待定/不纳入：head基身份不应随意消除，源实验也涉及AFEW。|
|18|[Neural-Kernel CME](https://proceedings.mlr.press/v235/shimizu24a.html)|待定／不计数|待定/不纳入：核心核目标与支撑结构不符合当前边界。|
|19|[KAN完整函数复合](https://arxiv.org/abs/2404.19756)|暂留设计审查|暂留：逐边一元函数与嵌套求和；不是换激活，也不是旧可逆coupling。|
|20|[Deep Lattice](https://arxiv.org/abs/1709.06680)|暂留设计审查|暂留：校准＋多维查表插值＋再复合；不假设情感单调性。|
|21|[可微逻辑算子电路](https://arxiv.org/abs/2210.08277)|暂留设计审查|暂留但逻辑家族邻近：固定布线选择真值表，区别NLM量词链。|
|22|[Dempster–Shafer原型层](https://arxiv.org/abs/2103.13549)|待定／不计数|待定：作者代码Ω合成与标准公式不一致，不能直接移植。|
|23|[高阶FM](https://arxiv.org/abs/1607.07195)|排除|排除：仍属已撤回的低秩乘法交互，代码归属也未闭合。|
|24|[Tensor Train](https://arxiv.org/abs/1509.06569)|排除|排除：压缩普通权重矩阵，不是新的证据处理链。|
|25|[GDN](https://arxiv.org/abs/1511.06281)|排除|排除：只搬归一化太小；保留完整密度学习又需新目标。|
|26|[IterNorm](https://arxiv.org/abs/1904.03441)|排除|排除：归一化旧方向且源代码依赖batch统计。|
|27|[FSQ](https://arxiv.org/abs/2309.15505)|排除|排除：量化算子加普通编解码器不满足完整Block门槛。|
|28|[Vector Neurons](https://arxiv.org/abs/2104.12229)|排除|排除：OSRAM隐坐标没有已知SO(3)旋转语义。|
|29|[完整HyperNetwork](https://arxiv.org/abs/1609.09106)|待定／不计数|待定：源动态LSTM是缩放，完整动态矩阵迁移尚未闭合。|
|30|[Dynamic Filter Network](https://arxiv.org/abs/1605.09673)|排除|排除：缺局部空间支撑；改全连接后与HyperNetwork合并。|
|31|[Neural ODE](https://arxiv.org/abs/1806.07366)|暂留设计审查|暂留：有限时间初值流，区别DEQ平衡点，数值时间非历史时间。|
|32|[Liquid Time-constant](https://arxiv.org/abs/2006.04439)|待定／不计数|待定：与ODE同家族，完整电导动力学不能裁成gate。|
|33|[N-BEATS generic](https://arxiv.org/abs/1905.10437)|待定／不计数|待定：双backcast/forecast链有差异，但可能仍只是残差MLP。|
|34|[Test-Time Training](https://proceedings.mlr.press/v119/sun20b.html)|排除|排除：要求新增自监督目标/头与推理参数更新。|
|35|[Differentiable Plasticity](https://proceedings.mlr.press/v80/miconi18a.html)|排除|排除：典型价值依赖新增可写关联状态，越过当前Memory边界。|
|36|[Predictive Coding](https://doi.org/10.1162/NECO_a_00949)|排除|排除：完整机制改变信用分配/局部学习，不是本次读出模块。|
|37|[Deep Feedback Control](https://arxiv.org/abs/2106.07887)|排除|排除：改变训练更新算法，不能测试时用未知target驱动。|
|38|[B-cos](https://arxiv.org/abs/2205.10268)|待定／不计数|待定：作者代码已读，主方法全文未闭合，且需排除cos gate退化。|
|39|[Janossy](https://arxiv.org/abs/1811.01900)|待定／不计数|待定：完整排列平均有区别，但固定语义槽是否需要该不变性未明。|
|40|[Matrix-Tree结构注意力](https://aclanthology.org/Q18-1005/)|待定／不计数|待定：树边缘不同于softmax，但仍属旧交互方向，单父先验无依据。|

## 暂留的9条到底哪里不同

|线索|保留的完整核心|最接近旧方法|尚未通过的关键条件|
|---|---|---|---|
|DPP|联合子集行列式概率→子集非线性表示→期望读出|ToMe、OTKE、Slot|多样性是否真有用；不得仅用边缘概率做gate|
|双曲网络|基点相关exp/log、Möbius映射与平移复合|SPDNet、Spline|没有证据支持读值层级；接近球边界的数值稳定性|
|BernNet|固定语义图的完整Bernstein谱响应|Sheaf、ED-HNN|与既有图传播的方向相近，不能先称不撞|
|图Scattering|固定wavelet bank→多阶模值路径→各阶低通摘要|Sheaf、本批BernNet|与BernNet共享图谱假设；极性信息可能被模值削弱|
|Sandwich|Cayley耦合权重＋正对角变量的整段前馈参数化|收缩DEQ|只能约束新分支；不能推出原Flat或跨mask整体更稳定|
|KAN|逐边函数→求和→函数复合|旧Spline coupling|必须完整非可逆函数网络，不是换激活或多堆层|
|Deep Lattice|校准→小维度插值表集合→再次校准/插值|NODE|不能给无语义latent强加单调约束；高维成本|
|逻辑算子电路|固定稀疏布线→16真值表松弛→多层组合|NLM、RAT-SPN|仍邻近逻辑方向；软训练/硬评估差异、连接覆盖|
|Neural ODE|初值状态→误差控制的有限时间积分→终点表示|DEQ、EA|只换Euler残差层不成立；批次共享步长和伴随误差|

这些不是9个完全无亲缘的大方向。尤其BernNet/Scattering属于同一谱图大类；逻辑电路与旧NLM仍有方向邻近；ODE与旧隐式迭代相邻。若用户的“不撞方向”是按大类而非核心算法定义，必须进一步合并/排除，不能把9当作最终合格数。

## 来源与代码缺口的具体例子

- OptNet：真实论文与QPFunction确实存在，但没有来自本任务的有意义耦合约束，KKT隐式反传本身不能绕过旧DEQ/EA去重，暂不接纳。
- PersLay：作者层处理已经计算好的diagrams；对diagrams可微不等于当前特征→拓扑配对→读出全链已可微。
- Dempster–Shafer：所读作者master的DS3_Dempster对包括Ω的整个向量使用三个相加项，使Ω也累加三次乘积；与标准singleton+Ω组合不同。需要固定修订或明确按论文重写验证，当前不能借“官方代码”背书。
- B-cos：代码已读，来源方法全文获取不完整；维持pending，不依据摘要补造原机制。
- HRR：固定key绑定/解绑可能整个退化为线性算子；增加名字和FFT不证明新机制。
- FSQ、Tensor Train、HOFM：有真实出处也不满足用户“不用简单算子或低秩包装凑数”的门槛。

## 留下的下一步，而非自动启动

只推进源码/数学适配仍有希望的设计缺口；没有让任何pending条目进入队列。若只能通过放宽“改loss/改Memory/新支持数据”的边界才能保留，先标为越界，而不是静默改变任务。9条暂留需要明确机制差异、固定计算图、mask/空历史语义与成本后才能成为实验候选。未找到40合格项是本次实际缺口，不说明数学上不存在40项。

[统一元数据](osram_next40_paper_bank.json) 包含40个唯一ID、作者数组、论文URL、源码URL、最近旧方法和状态；详细卡片在 [优化](osram_next40_optimization.json)、[几何](osram_next40_geometry.json)、[表示](osram_next40_representation.json)、[条件计算](osram_next40_conditional.json)、[补充](osram_next40_supplement.json)。所有收益是迁移假设；本次没有新性能数字。按文献溯源流程保存拒绝原因，避免下轮重复搜索已经否决的包装。

代码基准：`42d4904`，工作分支 `feature/osram-uniform-forced-text`。只新增文献记录，不修改运行源码、已接受方法注册表、checkpoint或远程队列。
