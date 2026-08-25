# 上下文快照（中文）

本任务只在 GCNet temporal 与 speaker 两个第二层 GraphConv 实现来源忠实的 GenAgg 和带尺度修正的 Soft Medoid。其他模型和协议组件全部不变，继承已有 Original 40个归档，不跑 epoch smoke，验证 Torch 1.8/PyG 2.0.1 兼容性，并在 IEMOCAPSix fold 5 运行12任务配对门控。完整主快照见[这里](second-graph-aggregators-20260825T140248Z.md)。

