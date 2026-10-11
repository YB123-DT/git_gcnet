# 固定特征 OSRAM：文献筛选
日期：2026-10-11。基线代码记录：8ef92c7。仅文献调研，无新训练或模型修改。

## 边界与结论
用户拒绝更换上游原始特征提取器。本轮不再把五槽图、标量门或简单残差换名字当新方向。以下都是跨领域迁移假设，未证明 MOSI 收益，也不构成新方法创新证明。
保留 TabM 为强 Flat 工程候选；有条件保留 TabR 为训练实例检索方向。RealMLP 仅作训练参考；Hyper-Connections 不推荐当前使用。

## TabM：改善读出学习，而非增加证据交互
[论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)；
[官方参考实现](https://github.com/yandex-research/tabm/blob/main/tabm_reference.py)；
[损失及推理](https://github.com/yandex-research/tabm/blob/main/paper/bin/model.py)。

大矩阵 W 共享，成员有独立输入/输出缩放和偏置：s_j * W(r_j * x)+b_j。多个成员分别接受任务监督，再平均预测；训练不是对平均预测算一次 loss。不是输入条件 Gate。
迁移设想：Memory 仅扫描一次，成员差异在首次 Local/Base/Gap 融合前引入，保留 Flat 型路径；MOSI 平均回归输出，分类平均概率。
风险：读出计算与激活随成员数增加；不是只复制末端 head。原论文附录 A.4 未研究 normalization backbone，OSRAM 的 LN/Skip 属迁移改造。
去重：旧 docs/osram_survey80_20261004/representation.json 已提过 TabM，因当时排除 individual-head loss 而 not_counted；不能称从未见过或已跑过。本轮没有授权解除该限制。
核查深度：论文方法/限制及官方模型、训练代码。

## TabR：引入参数外的训练实例证据
[论文 v2](https://arxiv.org/pdf/2307.14338)；
[官方实现固定版本](https://github.com/yandex-research/tabular-dl-tabr/blob/17baa9082506f8e7a0f8d11bb1e08212926a1507/bin/tabr.py#L235-L312)。

k=W_K x；邻居权重由负平方 key 距离 softmax 得到；value=训练标签编码+T(k-k_i)，加权后残差注入。eval 的目标标签不传入；候选仅来自训练集。
迁移设想：用合法 L/B/G/a 查询其他训练对话，不写入 OSRAM persistent memory。区别在于新增样本证据来源，而非仅改变五槽交互。不能声称恢复目标缺失 Text。
必须新增整段 conversation 排除（原代码仅排 self）；每 fold 单独训练库；禁止测试标签入库。缺失模式不匹配、错误邻居、在线主干导致库陈旧、重编码成本是主要风险。不是轻量即插即用，且仍使用残差。
核查深度：arXiv v2 正文/附录与官方完整核心文件。最终会议 PDF 访问受限，未冒称核对最终版本。仓库无 TabR 名称命中不等于已排除所有等价实现。

## RealMLP：训练配方参考，不是新 Block
[NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)；
[作者代码](https://github.com/dholzmueller/pytabkit)。
重点是预处理、参数化、初始化和训练配方组合，不只是扩大 MLP。原论文跨数据集验证不等于适用于训练中不断变化的 L/B/G。仅核对论文及代码入口，未完成实现审计，不推荐直接整套移植。

## Hyper-Connections：当前淘汰
[ICLR 2025 论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf)。
扩展残差流，学习层间 width/depth mixing；核对正文和附录 J 作者 PyTorch 伪代码。主要面向深层网络的传播，不足以解释当前浅 Flat 的问题；不为使用它而人为加深网络。第三方 lucidrains 库不当作作者实现。

## 决策
追求先改善现有强 MLP：继续评估 TabM 是否允许改变成员监督。
追求不同的信息来源：进一步评估 TabR 的缺失条件检索与训练库成本。
二者均未获实现/启动授权；不把工程收益候选包装成论文创新。已有运行不受本次调研影响。
