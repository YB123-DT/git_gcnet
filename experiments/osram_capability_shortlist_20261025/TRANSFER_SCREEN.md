# 知识传递与学习阶段：第二轮筛选

研究基点33e002e。只读调研，无新模型、训练或数据修改。

## CMAD（ICCV2025）：优先深读，不能原样复制

论文：CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities。
作者仓库 https://github.com/YetZzzzzz/CMAD ，核查HEAD efc812bee1549922528179cc2e94bfa066a3b2b6。
官方论文检索可获取摘要/段落，但直接全文请求403；不宣称全文消融核验完成。

实际读取Student_Model/loss.py及train_student_model_mosei.py：
- CAFD结合逐样本特征匹配与student-teacher、teacher-teacher跨样本相似结构；区别于简单同样本对齐，但有关系蒸馏近邻。
- combination_importance按组合聚合relu(student task loss - teacher task loss)，按最大值归一化再平方。
- 公开MOSEI训练脚本：begin_epoch之前实际权重为全1；在begin_epoch从累计训练记录计算权重；之后保持不变。不能称为每轮持续动态课程。
- 权重同时作用于任务L1和输出蒸馏MSE，不只是蒸馏项。不是用户原MSE协议。
- regression_loss中的temperature在标量MSE公式中代数抵消，不能照搬分类温度解释。
- max归一化与correlation比率可有零分母，移植需显式测试。
- 需要教师训练成本、同预算双视图任务控制、均匀KD和组权重消融。保持GCNet mask，学生不可输入训练特权模态到原写入。
- 教师优势大并不保证缺失输入可学；难条件可能存在不可约误差。迁移机制未证明。

## G2D（ICCV2025）：保留，但实现适配成本更高

作者仓库 https://github.com/rAIson-Lab/G2D ，核查HEAD 27318d47de32c0b970d45d6ce6071a05b54de97f。
论文 https://arxiv.org/html/2506.21514v3
读取student.py与utils/utils.py::modulate_gradients_based_on_scores_from_two_modalities。
独立Audio/Visual教师提供特征及输出监督；训练中有顺序模态优先更新。
当前公开函数以教师正确类别概率和比较两个模态，设置梯度系数0/1；仅作用于顶层名称含audio/visual且参数为4维的张量。直接复制到OSRAM Linear投影不会作用。
公开入口是两模态CE分类，不等于现成MOSI三模态回归实现。需回归适配、有效模态mask、共享历史梯度范围及额外教师预算控制。已有配对诊断不支持把普遍梯度冲突当项目事实；G2D的模态能力保存也不等于梯度冲突假说。

## MCKD：暂不优先

阅读arXiv v1 https://arxiv.org/html/2405.07155v1 ，版本题名Enhancing Multi-modal Learning: Meta-learned Cross-modal Knowledge Distillation for Handling Missing Modalities。
使用独立meta-validation学习模态权重，并以权重比调整跨模态特征蒸馏。原结构还用可见模态平均特征填充缺失槽，不能整套迁移后声称保留真实观测原则。需要双层优化和任务适配；没有验证当前情感任务迁移。版本题名可能变化，记录v1而非混同最新版本。

## 历史去重的重要修正

experiments/osram_supervised_teacher_20260914/RESULT.md明确说仅工程验证：没有完整Teacher预训练/五种子Student。固定教师只导出ObservedSet projectors，并沿用JEPA SmoothL1/InfoNCE目标；不能归类为“完整任务蒸馏已失败”。
paired_history_views_20261001是两视图任务监督与额外InfoNCE对照，不等价于冻结任务教师蒸馏。
r18_lupi.py用完整训练特征控制乘性高斯噪声，不是教师任务知识迁移。
上述是局部核查，不保证整个仓库没有其他蒸馏实现。

## 当前建议

先把CMAD的关系蒸馏、条件权重分开评估设计；G2D作为另一教师方向候选。不是直接启动二者，也不是将论文方法改名为OSRAM创新。原论文的等预算证据尚未完整核验，不声称已解决预算混淆。

