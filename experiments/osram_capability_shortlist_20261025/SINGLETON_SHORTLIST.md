# 单模态有效性筛选：继续检索，尚未选定实现

2026-10-25；仓库基准 4643a97；仅文献与代码审阅，没有新增训练。
采用 Literature Trace 与 Idea Convergence：保留两个候选，等待用户选择，不自动实施。

## 新的筛选边界

GCNet random-mask-v1 的 rate=.7 当前仅一个模态。候选必须在单模态时仍执行其核心运算；“可以执行”不代表能恢复不可观测信息或必然提分。保持现有固定话语特征，不要求新的原始音视频输入。

## 排除重复

- ReconBoost 的先前 missing-aware 改写在 singleton 时退化为普通任务损失，暂不作为高缺失主方案。
- 已有 TRAIN exemplar 检索不是空白：osram_exemplar_switch_20261023/RESULT.md 中 label-aware ON 相对原 Flat 八率 W-F1 +0.006pp、高缺失 -0.391pp；这只是该版本、该协议结果，不是全部检索方法无效。
- PCD 论文具有概率对齐区别，但作者仓库 README 实际仍为 Code is coming soon，不能当成已核对完整开源实现。又与已有教师/概率表示路线相近，本轮降级。
- 通用 MoE、Target-private PLE 已有。后者位于 missing_m3_target_ple_20260903，针对 JEPA 缺失目标私有专家，不等同于当前任务表征的缺失模式路由；仍必须承认概念重叠。

## 候选1：MARS 式特权偏移指导专家分工

论文：https://arxiv.org/abs/2606.30355
代码：https://github.com/seunghub/MARS
审阅源码版本：f94cf9337082a763452974cf9aac3fd7a0dbb73b。
核心文件：CASIA-SURF/classification/models/our_model.py。

论文将完整/缺失表征的差用于训练路由器；专家本身处理缺失表征。另一个仅看缺失表征的路由器学习该分配，供推理使用。其目的不是预测缺失 Text 向量，也不是三个固定预测候选加权。

实际源码确认：NoisyGate 在 residual=None 时只用 feature router；训练 KL 目标对 residual-router logits detach；另有负载与噪声相关机制。论文写 full-partial，所审代码写 partial-full，迁移必须固定符号。该分类实现计算所有 expert 后加权，不能凭 top-k 宣称稀疏算力节省；eval 仍计算未用于最终路由的完整 anchor，迁移应移除这个无用路径并验证预测无依赖。代码自带缺失采样与自适应概率不能替换 GCNet sampler。

拟议迁移（尚未实现、不是原论文实验）：将任务侧 Local256 送入多个非线性专家，输出仍为256维；再与未经改变的 Base/Gap 进入原 Flat。训练时额外计算同一句完整当前输入的 Local 供特权路由监督，绝不写入主 CHM。原缺失 Local 继续构造 CHM Key/Value/Query，Memory 单次扫描；任务侧 Local Skip 是否使用专家输出必须在实施前固定，两组完全一致，不能实验中变更。单模态 A 时，专家和部署路由器均只看 A 所形成的 Local；完整输入只出现在训练监督分支。

最小对照：相同专家、输入、容量、预算，普通 feature-router MoE vs 特权偏移指导；Flat 为既有同协议参照。必须观察部署路由器结果，不能把训练特权路由器结果当成绩。若不优于普通 MoE，不能归因于偏移指导。MOSI 回归与固定特征迁移尚无验证；完整模态差未必能由单模态推断，是主要失败风险。无声称已有优化冲突证据。

## 候选2：MaMOL 式缺失条件参数专用化

论文：https://arxiv.org/abs/2511.11460
代码：https://github.com/Jiahuiqu/mamol
审阅版本：6d7dd482f8302d12b34a36d76976ec7013aa19ea。
核心文件：momal_3modality/modules/vision_transformer_moe_3.py。

论文采用共享/专用轻量专家与缺失条件路由。迁移假设是 A-only 与 AT 不必使用完全相同的任务变换，而不是重新给 Base/Gap 乘权重。在当前仅 A 时依然有专用计算，不依赖模态两两同时存在。

这不是直接可复制代码：审阅文件把 missing bits 先线性映射成 embedding，再以 embedding 的 >0.5 阈值启用静态专家，不能把它解释为精确模态 bit；None 检查也发生在 as_tensor 之后。需要另行审计论文与各入口，不能原样接入。与原 PLE、Soft MoE 有明确家族重叠，创新优先级低于候选1。

最小迁移对照：容量匹配的共享 adapter vs 由真实 availability 条件化的 adapter；保持相同输入位置及冻结/训练设置。训练条件之间的梯度冲突尚未证明，不以此预设提分原因。该方案可作为参数分工能力检查，不足以单独宣称新方法。

## 本轮结论

优先进一步讨论候选1，候选2为更保守的比较路线；不是批准实验或宣称二者有效。没有改模型、mask、环境、训练任务和历史结果。
