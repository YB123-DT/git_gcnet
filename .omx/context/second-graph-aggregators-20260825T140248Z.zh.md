# 上下文快照（中文）

本任务在 GCNet 图主干中实现并评估 GenAgg、带尺度修正的 Soft Medoid、SSMA，以及明确标注为自定义假设的 RTDR relation-transition routing。继承已有 Original 40个归档，不跑 epoch smoke，并验证 Torch 1.8/PyG 2.0.1 兼容性。纯线性RTLF、与现有GraphConv代数冗余的Ego–Neighbor Separation、无法忠实迁移的Centered Clipping只保留否决证据。分两波运行共24个候选任务，完成校验后自动上传代码与结果。完整主快照见[这里](second-graph-aggregators-20260825T140248Z.md)。
