# 测试规格（中文）

验证 legacy 路径精确一致；手算 GenAgg、Soft Medoid 与 SSMA 机制；梯度有限、参数数正确、集成边界受控；训练、CLI与归档身份不碰撞；resume不可变；官方训练环境只做一次GPU检查。RTDR只有在Original原路径bit-exact、full-transition按容差等价后才能运行diagonal。Ego代数冗余与Centered Clipping不可迁移只形成证据。两波各12任务，共24个候选任务，完成后校验并自动上传。完整主测试规格见[这里](test-spec-second-graph-aggregators.md)。
