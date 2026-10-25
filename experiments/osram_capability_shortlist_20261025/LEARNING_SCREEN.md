# 学习机制检索：先检查已有反证

本次只读取现有结果与代码，没有新增训练或诊断。

## 关键本地证据

`experiments/osram_text_condition_optimization_audit_20261022/RESULT.md` 已完成 MOSI/MOSEI 各64个训练目标的 Text 可见/缺失配对。固定严格过去，eval关闭dropout。总任务梯度平均余弦0.426/0.584，负向16/11对恰好对应标量MSE残差异号。预测Jacobian整体余弦0.861/0.868，没有负向对。单步SGD 1e-5可见→缺失平均ΔMSE -0.024924/-0.003843；改善47/53对。这不支持把普遍条件梯度冲突预先认定为瓶颈。

边界：旧s0环境、每数据集一个seed66 miss0.3 BEST权重、有限单目标样本；不是当前cmerc训练全过程、跨样本或Adam长期更新结论。不要因此宣布一切优化干扰不存在，也不要重复运行同一诊断。

`gcnet_missing_m3/core20.py:201-205,228-246` 表明已有C20 CAGrad针对local/base/full三个累计出口损失；不是按A/T/V/AT/AV/TV/ATV分组。更换分组有实验区别，但不是新的优化算法。

## 文献筛选

MLA（CVPR2024）是交替单模态更新、共享头梯度保护与推理融合。MMPareto（ICML2024）处理单模态辅助目标与多模态目标冲突。CAGrad（NeurIPS2021）是多目标更新基准。三者的任务定义和参数作用范围不同，不能仅因为均谈冲突就视为适合当前OSRAM。当前暂停优先推荐梯度保护，不启动训练。

后续检索应按机制/作用参数/新增监督/历史近邻/预算对照筛选，尤其区分条件知识迁移、学习阶段调度和冲突处理。可见条件目标更有信息不代表每个样本更准确，不能强迫缺失输入拟合不可推断的目标。

来源：
- https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html
- https://proceedings.mlr.press/v235/wei24d.html
- https://proceedings.nips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

