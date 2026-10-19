# MOSI Router：实现完成，用户要求暂不启动

用户新增一组MOSI版本，对应原Flat150三种子均值80.747的模型配置；
seed66历史Test-best为81.068095，不能误当80.7单种子。

已写好标量Router、严格数值桥接、150轮训练和有限队列。主干/数据/原
conversation-mask-v2沿用；Softmax只用于三个路由权重，不作用于标量预测。
训练和评价沿本轮单一validation选点规则，旧Test-best只是历史参考；
因此新增同配置Flat对照，不把不同选点口径混作模块增量。

当前 **NOT STARTED / PAUSED BY USER BEFORE LAUNCH**。
SSH已恢复，用户随后明确要求先别启动，因此停止后续部署和启动；不自动恢复。
只通过本地单元/梯度/恢复/元数据/LR/指标与队列测试；没有真实MOSI GPU smoke、
数值桥接结果或新训练成绩。不得把编写了桥接检查误称为真实数据已通过。

仅在用户重新要求启动后：核对健康GPU与原s0环境，固定当前源码snapshot，运行orchestrate入口。
必须先通过旧保护checkpoint的精确预测/标签/mask重放和原八率mask SHA检查，
随后真实batch梯度与optimizer smoke。只有通过才启动seed66 Flat/Router两组。
失败则记录并停止MOSI队列，不擅自放宽精度检查或换基线。

所有旧权重、缺失mask与实验记录保持未改动；未迁移到本地或其他服务器。
