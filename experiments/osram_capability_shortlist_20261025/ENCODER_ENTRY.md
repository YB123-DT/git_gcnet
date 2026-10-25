# 编码入口候选：论文溯源

仅文献初筛；未启动训练，未逐文件审计作者实现。官方论文与作者仓库入口已核对。

## MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition (ICLR 2026)

[论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7038033f576a57a1c8469f2c6c280fcc-Abstract-Conference.html) · [作者代码](https://github.com/Tree-Shu-Zhao/MoRA)

## Decoupling Common and Unique Representations for Multimodal Self-supervised Learning (ECCV 2024)

[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04236.pdf) · [作者代码](https://github.com/zhu-xlab/DeCUR)

## Boosting Multimodal Learning via Disentangled Gradient Learning (ICCV 2025)

[论文](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Boosting_Multimodal_Learning_via_Disentangled_Gradient_Learning_ICCV_2025_paper.html) · [作者代码](https://github.com/shicaiwei123/ICCV2025-GDL)

MoRA：共享模态公共适配参数，并保留模态专属适配；适用于预训练视觉语言模型，不能直接等同在现有全可训练投影前加低秩线性层。与用户过去低秩双线性融合不同，但仍需核对参数化是否有额外能力。

DeCUR：共同维度跨模态相关矩阵趋于单位阵，独有维度跨模态相关趋于零；模态内双增强学习约束全部维度，防止独有部分退化。不可只移植拆分结构而去掉训练目标。与FactorCL等共享/独有表示家族有概念重叠，不声明全新研究问题。

DGL：融合目标与单模态目标按参数职责隔离梯度。它是优化方法，不包装为推理第二Block。原论文含MOSI三模态实验，但不是当前GCNet缺失/特征/CHM协议。

检索排除：SyP官方页写代码将发布，未据此声称已核对完整开源实现；CoMM与既有对比训练方向近；ReMix跨样本重组不能直接用于因果对话历史。

仓库命名检索：在experiments的md和paper-bank JSON中未找到DeCUR/两个指定仓库URL；不等于排除所有概念重复。

下一步仅深读消融及代码，分别检验输入功能、训练约束和历史工作去重；没有新增实验授权。

