# MOSI Router：已获重新启动授权，等待资源及真实检查

用户新增一组MOSI版本，对应原Flat150三种子均值80.747的模型配置；
seed66历史Test-best为81.068095，不能误当80.7单种子。

已写好标量Router、严格数值桥接、150轮训练和有限队列。主干/数据/原
conversation-mask-v2沿用；Softmax只用于三个路由权重，不作用于标量预测。
训练和评价沿本轮单一validation选点规则，旧Test-best只是历史参考；
因此新增同配置Flat对照，不把不同选点口径混作模块增量。

当前 **MODEL NOT STARTED / WAITING FOR RAM ADMISSION**。
用户暂停后已明确重新要求“启动”。2026-10-19 14:03UTC服务器RAM仅约1.6GiB
可用，GPU空闲不代表可以训练。由轻量资源队列等待可用RAM至少64GiB后准入，
仍需完成真实数据数值桥接及smoke，不提前声称已训练。
本地部署恢复进程已运行，先等待SSH；共同启动记录见
`../osram_text_path_router_3seed_20261019/RESUME_LAUNCH.json`，不要重复启动同一计划。
只通过本地单元/梯度/恢复/元数据/LR/指标与队列测试；没有真实MOSI GPU smoke、
数值桥接结果或新训练成绩。不得把编写了桥接检查误称为真实数据已通过。

获得RAM准入后：核对健康GPU与原s0环境，固定当前源码snapshot，运行orchestrate入口。
必须先通过旧保护checkpoint的精确预测/标签/mask重放和原八率mask SHA检查，
随后真实batch梯度与optimizer smoke。只有通过才启动seed66 Flat/Router两组。
失败则记录并停止MOSI队列，不擅自放宽精度检查或换基线。

所有旧权重、缺失mask与实验记录保持未改动；未迁移到本地或其他服务器。
