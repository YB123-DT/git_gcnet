# MOSI Router：资源已准入，preflight因历史源码快照缺失停止

## 最新实测：2026-10-19 14:30–14:34 UTC

资源恢复后，独立队列PID2742102在物理GPU7启动smoke子进程2742105。
子进程在读取历史源码`SNAPSHOT.json`时退出1，尚未执行真实batch的
forward/backward，Flat/Joint150均未启动。没有模型性能结果。

缺失路径：
`/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0/SNAPSHOT.json`。
整个source目录不存在。原数据DATA.json和受保护Flat seed66权重目录仍存在。
对/data2和/data1实验目录中三层内的SNAPSHOT.json检查，未找到code_commit
为ad211c0的可复用快照；未绕过校验、伪造旧manifest、改基线或反复盲目提交。

原始失败日志保留：
`/home/qinjinghui/remote_experiments/osram_text_path_router_joint_mosi_20261019/execution_v1/smoke/joint.log`。
后续须找回/可审计重建历史源码来源并通过精确重放，再使用独立运行目录。
MELD/IEMOCAP八组不受此失败影响，继续正常训练。下文等待资源是先前状态记录。

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
