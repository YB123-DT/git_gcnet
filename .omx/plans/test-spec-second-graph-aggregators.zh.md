# 测试规格（中文）

验证 legacy add 精确一致；手算 GenAgg 与 Soft Medoid；梯度有限；参数数正确；只替换两个第二层；训练 loss 路由正确；CLI与归档身份不碰撞；runner 在 `stage=formal` 恰好构建12个任务且证明gate不能冒充formal；resume不可变；历史Original仅允许新增字段缺失并映射legacy默认值；配对门槛正确；官方训练环境只做一次GPU检查；完整回归通过；输出双语证据。完整主测试规格见[这里](test-spec-second-graph-aggregators.md)。
