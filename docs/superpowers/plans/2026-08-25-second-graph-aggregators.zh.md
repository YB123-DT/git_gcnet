# GCNet 第二层聚合器实现计划（中文镜像）

完整逐步计划见[主文件](2026-08-25-second-graph-aggregators.md)。计划包含八个TDD任务：数学单测与兼容实现；legacy精确接入；训练/CLI/归档身份；复用锁定runner；继承Original汇总；完整本地验证与一次官方GPU检查；四卡12任务判别；按门槛自动继续或停止。全过程不重跑Original、不重复首批任务、不跑epoch smoke。

