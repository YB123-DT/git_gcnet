# GCNet 新增图候选决策（中文）

主文档：[2026-08-25-additional-graph-candidates-design.md](2026-08-25-additional-graph-candidates-design.md)。

## 结论

- SSMA Conv2：接受。来源是 NeurIPS 2024 的 Sequential Signal Mixing Aggregation，确实用循环卷积中的跨邻居乘积解决sum无法在压缩前建模邻居交互的问题。
- Relation-Track MMP：原名称和“纯延迟融合”均拒绝。检索到的MrMP每层都会立即求和relation；而在线性第二层中，保留全部二跳relation组合时延迟求和与提前求和严格等价。非平凡实验改名为自定义 `Relation-Track Diagonal Routing (RTDR)`，明确改变的是二跳relation transition mask，不得包装成论文迁移。
- Ego–Neighbor Separation：正式候选拒绝。GCNet已有独立 `lin_l` 邻居矩阵和 `lin_r` 自身矩阵，显式self edge只造成可重参数化的重复；立即concat再linear与现有函数类等价。
- Centered Clipping：拒绝/暂停。原方法允许任意初始中心，官方实现默认使用上一轮聚合；但迁移到GCNet时，中心初始化、迭代次数、阈值尺度以及“好点/异常点”语义都没有依据。其理论还要求异常比例不超过约10%；GCNet邻域只有3–5条边，一个异常点已占20%–33%。

## SSMA正式适配

论文与证据：[NeurIPS论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html)、[arXiv](https://arxiv.org/abs/2409.19414)、[检查的官方代码commit](https://github.com/AlmogDavid/SSMA/tree/9d128c902acf47343b5baf5150a78dfb6a64fb3e)。仓库README声称MIT但缺少LICENSE文件，因此只依据公式独立实现。

锁定formal维度 `d=100`、最大入度 `kappa=5`，所以 `m1=6,m2=496,m=2976`。固定系数信号令第一行前100项为 `-message`、第二行第一项为1，其余为0。对每条邻居信号做FFT2；邻居间对log幅度取mean再exp，对相位求和；polar重建后IFFT2取实部；最后用论文base full compressor `Linear(2976,100)` 压回100维。

GCNet中把SSMA作为 `GraphConv` 的聚合器：先对原始邻居状态做SSMA并用论文base full compressor压缩，再经过现有 `lin_l`；`lin_r(root)` 只加一次。这个顺序与PyG `GraphConv(aggr=...)` 一致，确保替换的是聚合而不是偷偷改变消息变换。保留全部edge和显式self edge，不加attention、不采样、不改conv1。每分支新增297,700参数，两个分支共595,400；新增初始化使用RNG分叉。正式训练前只在biggpu官方Torch1.8环境做一次complex FFT/polar backward检查。

## RTDR实验性适配

若保留所有二跳组合，线性关系轨满足 `sum_q A_q sum_r mu_r = sum_{q,r} A_q mu_r`，所以纯RTLF必然与Original相同。RTDR是明确标注的零新增参数自定义假设：Original保留全部 `(q,r)` transition，RTDR只保留对角项 `q=r`。`A_q` 是对 `edge_type=q`（含其self edge）的精确未归一化add-scatter，`A=sum_q A_q`。第一层root/bias作为common track走完整原邻接，conv2 root作用于完整融合状态一次，bias只加一次。`early`原路径必须bit-exact；因分解会改变求和顺序，`full-transition`按前向 `1e-6`、反向 `1e-5` 容差验证；之后`diagonal`才可训练。论文中不能写成MrMP迁移或仅延迟融合。

## 实验任务

可训练候选为 GenAgg、Soft Medoid、SSMA、RTDR。每个候选使用missing `{0.0,0.7}` × seeds `{66,67,68}` 的6个formal子集任务，共24个新任务：Phase A 是已经注册的 GenAgg + Soft Medoid 12任务；Phase B 在SSMA和RTDR selector/runner通过测试后运行另12任务。四张卡每卡三个进程，不运行Original，且全部任务可直接继承进正式扩展。

SSMA若通过，还必须与参数匹配的sum+MLP控制比较后才能归因；RTDR只报告为自定义relation-transition routing机制。两个被拒绝项写入总账，不再重复调研或训练。

所有正式任务完成并通过本地provenance校验后，自动同步回 `/data2/yb/paper`，生成中英双语结果，按Lore提交代码与完成结果并推送到 `https://github.com/YB123-DT/git_gcnet`；失败或未完成任务不包装成完成版本上传。
