# 第二 Block：不同机制的初筛

仅完成方法段与关键实现核对，未声称逐页全文审查或移植验证。未改代码、未训练，尚未选定最终方向。原特征、GCNet missing和CHM保持现有约束。

| 候选 | 来源／保留机制 | 与历史实验的区别与风险 |
|---|---|---|
| MultiModN | 医疗等任务；按可见模态逐次更新任务状态，跳过缺失编码器 | 不同于Local→Base→Gap标量/logit修正；但与层级多出口监督相近，不能只改名。其主要缺失优势来自MNAR实验，不能承诺GCNet随机缺失涨分。 |
| TabM | 表格学习；共享大部分权重的多预测训练 | 不是MoE或多Memory轨迹；必须保留成员监督与预测聚合，不是把多个hidden平均后套一次head。参数节省不等于算力免费，迁移对任务头/损失接口有影响。 |
| Hyper-Connections | 大模型/视觉；跨层维持多条残差流并学习宽度/深度连接 | 不等于旧单路Gate；但与cross-stitch/Mixer相近，单层混合不足以构成原机制。浅Flat未证明存在其针对的深层训练问题，暂降级。 |

来源：[MultiModN](https://arxiv.org/html/2309.14118)、[TabM](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)、[Hyper-Connections](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf)。

实现边界：MultiModN公开代码对整个batch检查any-NaN后跳过，不能直接照搬到逐话语不同availability；还包含状态变化惩罚。TabM已核对LinearBatchEnsemble共享weight、成员r/s/bias以及论文的成员损失/聚合区别。Hyper-Connections只核对论文附录作者PyTorch式实现，未核实独立官方代码库。

去重范围：osram_method_registry.json、meaningful/core20实现、decision_correction.py、模态轨道/CED结果。未找到这三个机制的同名完整实现，不代表已穷尽所有历史别名或证明新颖性。旧Local/Base/Gap图交互、原型、普通门控和补全不重新包装。

建议用户在前两项中选择继续细化；第三项不优先。没有候选已被证明适合MELD。进入代码实现前须明确第二Block职责与任务头/损失是否允许变更；不会擅自扩大训练。
