# PRD：GCNet 第二层聚合器（中文）

实现来源忠实的 GenAgg 与带尺度修正的 Soft Medoid，且只替换两个第二层 GraphConv 聚合器；保证 legacy add 精确不变；继承已有 Original 证据；首轮只跑12个配对任务；仅让满足锁定门槛的候选晋级。完整主 PRD 见[这里](prd-second-graph-aggregators.md)。

