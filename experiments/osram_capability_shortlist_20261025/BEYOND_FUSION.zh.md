# 排除重复融合后的检索

审查基点：6330051。仅调研，没有训练、模型修改或完整实现审计。

边界：不重做特征提取器，不改变 CHM 内部，不把既有读出融合变体包装为新能力。

## 筛选结果

1. MLA，CVPR 2024：交替优化单模态预测函数，通过受梯度修改保护的共享头传递知识，推理时使用不确定性融合。官方论文明确描述这些机制。它是训练/预测范式变化，不是可直接插入 OSRAM 的独立 Block。迁移需验证共享头梯度保护与当前历史 BPTT 的兼容性；不能直接声称比已有 CAGrad、独立缺失率训练或双视图新。两个论文所列作者仓库 URL 在本轮网页工具均返回404，因此代码尚未核验。

2. MiDl，ICLR 2025：针对缺失模态的无标签测试时适配，以预测和可用模态之间的互信息及自蒸馏约束进行调整。作者仓库 README 可访问。当前未审计实现细节。与历史 MEMO 在测试时适配层面有重叠，尚未证明机制独立；改变评价协议且需要回归适配，暂不优先。

结论：本轮没有确认满足全部边界的独立第二 Block。MLA 可作为后续深读的训练机制，不是推荐立即训练。不能把未发现可行候选表述成所有位置均无解。当前表示变换不必要求重训上游提取器，但其新能力需要证明，不能仅因插入位置不同而成立。

## 来源

- MLA: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html
- MiDl: https://proceedings.iclr.cc/paper_files/paper/2025/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html
- MiDl code: https://github.com/meryusha/midl_tta

去重范围为现有 BLOCK_CATALOG 和局部源码检索，不等于完成所有历史实验的等价性审查。

