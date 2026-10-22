# 原论文代码的连接证据

只读审查本地已有代码；没有下载新模型、复现论文分数或改动这些仓库。
Git版本及引用文件SHA256另外保存在 `SOURCE_MANIFEST.json`。以下链接固定到所核对提交。

## MulT：序列交互位于汇总之前，但有多层残差

仓库：`/data2/yb/paper/msa/repos/Multimodal-Transformer`；提交 `a670936824ee722c8494fd98d204977a1d663c7a`。

- 输入是预提取的三模态特征序列，不是原波形/图像，也不是已经融合的OSRAM读值。[输入](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/dataset.py#L22)。
- 对文本目标，`Q=text,K/V=audio` 与 `Q=text,K/V=vision` 各输出一个文本长度的序列；两者concat，再进目标memory Transformer，最后取最后位置。三个目标分支再concat进入预测。[models.py:92](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L92)。
- Attention/FFN都有残差，末端MLP也 `last_hs_proj += last_hs`。不能说主路径中的交互消除了原特征通道。[Transformer:144](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/modules/transformer.py#L144)、[末端:141](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L141)。
- 实际训练只用最终任务criterion；返回hidden不接受另外的对齐监督。MOSI/MOSEI默认L1。[训练:159](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/train.py#L159)、[配置:147](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/main.py#L147)。
- 删除cross-attention会移除逐token跨模态条件化，但三个目标分支和残差可能仍保留原模态信息。不能声称删掉交互必然完全删掉某种模态。

## MISA：拆分后的六向量就是融合头的输入

仓库：`/data2/yb/paper/msa/repos/MISA`；提交 `ec42faddde0d210cf7368aebf2118fe9570e7102`。

- BERT masked mean、A/V两层双向RNN终态先形成句级向量，再投影到共同宽度。[编码路径:184](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/models.py#L184)。
- 每个模态的投影向量并行输入private encoder和跨模态共享的shared encoder。六向量stack后Transformer，再concat到主fusion；其token数6是表示分支数，不是六句话。[拆分:262](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/models.py#L262)、[融合:244](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/models.py#L244)。
- 默认总损失：MSE任务 +0.3差异 +1.0相似性 +1.0重构。差异约束针对同模态private/shared与跨模态private；相似性默认三对shared的CMD；重构由private+shared预测投影后的原句向量。[实际loss:118](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/solver.py#L118)、[操作数:328](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/solver.py#L328)。
- 没有把原三个未拆分向量直接旁接主head；Transformer内部仍有标准残差。重构结果不进入同次预测。`sp_discriminator`虽计算，其输出未进实际总loss；不能把声明的每个组件都算作有效监督。
- 对OSRAM的启示限于：新表征的定义、预测消费方式、训练约束是一套设计。不能将Local/Base/Gap直接当成三种原始模态套shared/private语义。

## ConFEDE：六个独立投影承担预测与分支监督

仓库：`/data2/yb/paper/msa/repos/ConFEDE`；提交 `640d8831b6c8befe2a438fe59671caee25ec4edd`。README明确先预训练编码器再训练融合，不将其描述为OSRAM式一阶段即插即用。

- MOSI文本BERT pooler，A/V Transformer后均值池化，先得到句级向量。每模态分别进入simi/dissimi两套投影，六套参数独立；每套LayerNorm→Linear768/384→Tanh→Dropout。[定义:13](https://github.com/XpastaX/ConFEDE/blob/640d8831b6c8befe2a438fe59671caee25ec4edd/MOSI/model/net/constrastive/TVA_fusion.py#L13)。
- 六向量concat为2304维后进入主head，原始编码器向量不直接旁接预测。训练还对这些分支使用共享mono decoder和对比loss。[拼接:91](https://github.com/XpastaX/ConFEDE/blob/640d8831b6c8befe2a438fe59671caee25ec4edd/MOSI/model/net/constrastive/TVA_fusion.py#L91)。
- 实际MOSI总loss是融合MSE +0.1 NTXent +0.01分支回归MSE；对比输入是六种投影输出及按相似度/标签选出的样本，训练器确实对 `all_loss` 反传。[loss:146](https://github.com/XpastaX/ConFEDE/blob/640d8831b6c8befe2a438fe59671caee25ec4edd/MOSI/model/net/constrastive/TVA_fusion.py#L146)、[训练:88](https://github.com/XpastaX/ConFEDE/blob/640d8831b6c8befe2a438fe59671caee25ec4edd/MOSI/train/constrastive/TVA_fusion_train.py#L88)。
- MOSI/MOSEI分支标签来自总体标签；SIMS的dissimi分支另用模态标签，不能混为同一监督设定。文本encoder初始冻结，代码中的epoch200解冻在默认25epochs内不触发。
- MOSEI名为weighted_NTXentLoss的类，默认没有调用 `update_label()`；类名不等于标签距离加权已生效。

## DLF：表示拆分和后续交互串联，并共同受监督

仓库：`/data2/yb/paper/msa/repos/DLF`；提交 `251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf`。

- 模态序列先投影、经三套specific Transformer和一套共享common Transformer，保留时间维。[编码:157](https://github.com/pwang322/DLF/blob/251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf/trains/singleTask/model/DLF.py#L157)。
- LFA将文本specific作Q，A/V specific作K/V；交互后经memory Transformer取末位置。共同空间另行自注意力和concat；最终将specific与common表示拼接预测。[LFA:264](https://github.com/pwang322/DLF/blob/251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf/trains/singleTask/model/DLF.py#L264)、[最终融合:305](https://github.com/pwang322/DLF/blob/251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf/trains/singleTask/model/DLF.py#L305)。
- 中间head、最终head、Transformer均有残差，输出权重没有整体零初始化。[显式残差:289](https://github.com/pwang322/DLF/blob/251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf/trains/singleTask/model/DLF.py#L289)。
- 总任务项包含final/common/T/A/V多个预测，文本specific权重3；另有重构、specific一致性、shared排序/对比及余弦约束。[实际损失:79](https://github.com/pwang322/DLF/blob/251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf/trains/singleTask/DLF.py#L79)。A/V specific预测已经过语言query条件化，不能再称纯单模态预测。
- `CosineEmbeddingLoss(target=-1,margin=0)`惩罚正余弦，并非严格强制正交。部分low-level logits未进入返回值和loss；构造的distillation kernel也不在实际训练model列表中。只报告真正使用的目标。

## Self-MM：主融合很普通，差别在共享表示的训练

仓库：`/data2/yb/paper/msa/repos/Self-MM`；提交 `1786283c81eeb507f317fa1c70a3faf77e67cee0`。

- BERT CLS和A/V LSTM终态得到句级向量；主预测是concat→MLP。单模态heads不接入M预测的数值路径。[主路径:54](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/models/multiTask/SELF_MM.py#L54)。
- 训练M/T/A/V四个目标。辅助标签由融合/单模态特征到正负中心的距离更新，辅助项使用 `tanh(|y_m-y_M|)` 加权L1。[训练:142](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/trains/multiTask/SELF_MM.py#L142)、[weighted_loss:236](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/trains/multiTask/SELF_MM.py#L236)。
- 伪标签/中心更新使用detached特征，但辅助预测损失仍可训练共享编码器。初始各分支标签相同，辅助权重为0，不能声称首步起所有分支就被强制优化。
- 固定已训练权重、eval时移除单模态heads不改变M预测；从训练中删去它们则失去相应监督。说明“新增组件必须是推理必经层”不是必要条件。

## MMIM：估计器连接在损失上，而不是主预测加法上

仓库：`/data2/yb/paper/msa/repos/Multimodal-Infomax`；提交 `cd0774c5a712ca5f1a5497dbf27dde11cade7434`。

- BERT CLS、A/V LSTM得到三句向量，concat送普通fusion MLP。MI估计器读取模态对；CPC读取融合表示和对应单模态表示。[模型:98](https://github.com/declare-lab/Multimodal-Infomax/blob/cd0774c5a712ca5f1a5497dbf27dde11cade7434/src/model.py#L98)、[实际模块](https://github.com/declare-lab/Multimodal-Infomax/blob/cd0774c5a712ca5f1a5497dbf27dde11cade7434/src/modules/encoders.py#L127)。
- 这些模块返回用于训练的标量，不把重构或互信息向量加到预测。阶段0优化MI estimator；阶段1优化任务 +αNCE −βlld，满足缓存条件时再−βH。[优化器:59](https://github.com/declare-lab/Multimodal-Infomax/blob/cd0774c5a712ca5f1a5497dbf27dde11cade7434/src/solver.py#L59)、[loss:182](https://github.com/declare-lab/Multimodal-Infomax/blob/cd0774c5a712ca5f1a5497dbf27dde11cade7434/src/solver.py#L182)。
- 不在当前optimizer中step与截断输入梯度不同；阶段1仍能通过估计器把约束传给编码器。`−βH`在 `if contrast` 外，关闭contrast不能直接当严格task-only。
- 特征/训练目标不同，不能把其结果当成“把InfoNCE再加到OSRAM就会涨”的依据；OSRAM已有双视图InfoNCE记录必须先遵守。

## ComP：明确的模态间信息交接，加分支任务监督

仓库：`/data2/yb/paper/05_reproduction/external_repos/ComP`；提交 `28192d3a5683543d7383e40898f9a98d1f114a08`。

- 输入先缺失置零。模态内shared/private编码后，audio shared拼接text/video prompt；随后第二轮prompt更新、第二段模态处理，再模态head和router联合head。[路径:200](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/model.py#L200)、[跨模态交接:243](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/model.py#L243)、[router:279](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/model.py#L279)。没有GNN。
- 第二阶段总loss含joint任务 +2×模态任务 +投影特征重构 +分解约束。重构target是**遮蔽输入的投影**，不是完整原始缺失模态。[训练:195](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/train_mr.py#L195)、[阶段loss:253](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/train_mr.py#L253)。
- Block内部与joint head前都有残差；删除prompt失去的是模态流里的其他模态通道，而最后joint head仍可融合模态。
- 原版prototype生成沿完整序列，attention也无因果三角mask。它与causal OSRAM不属于同信息可用范围；只改attention mask未必解决prototype的未来访问。
- 核验细节：训练器gradient-modulator按 `prop_former_a.proj_n.fc1` 等名字匹配，实际名字带 `_1/_2`，静态上不匹配。不能直接按说明文档声称当前提交已执行该调制。[匹配:276](https://github.com/WenjueHE/2026-AAAI-ComP/blob/28192d3a5683543d7383e40898f9a98d1f114a08/ComP/train_mr.py#L276)。本审查不修复他人代码，也不倒推论文成绩。

## 原GCNet：图表示接受分类和缺失重构，但原节点有旁路

本地 `/data2/yb/paper/GCNet_TPAMI` 的HEAD是 `6c7909056926b9e5b746366b80f3b27e75fe7db0`；所引用 `model/graph/module/train_gcnet/loss.py` 与upstream `f43d2788481fd5889148f08b688259c9fd712002` 相同。

- 缺失置零的concat经BiLSTM，再temporal/speaker双图。图内部 `concat(features, graph_output)` 保留pregraph表示；之后时序编码、两支相加。[GraphNetwork:50](https://github.com/CASIA-Affective-Computing-Group/GCNet/blob/f43d2788481fd5889148f08b688259c9fd712002/gcnet/model.py#L50)、[主模型:134](https://github.com/CASIA-Affective-Computing-Group/GCNet/blob/f43d2788481fd5889148f08b688259c9fd712002/gcnet/model.py#L134)。
- 分类与重构从同一个hidden分叉，重构损失只在缺失模态上对完整target计算。只有启用reccls才把补全再送一遍模型，默认False；不能把默认路径说成“先补全再分类”。[训练:337](https://github.com/CASIA-Affective-Computing-Group/GCNet/blob/f43d2788481fd5889148f08b688259c9fd712002/gcnet/train_gcnet.py#L337)、[重构loss:43](https://github.com/CASIA-Affective-Computing-Group/GCNet/blob/f43d2788481fd5889148f08b688259c9fd712002/gcnet/loss.py#L43)。
- 原版前后双向RNN、图未来窗口及可选全序列attention都有未来访问。不能把原论文/原代码成绩直接作为严格因果CHM的接法对照。

## SDR-GNN：图算子串联，但必须核对实际返回值

仓库：`/data2/yb/paper/SDR-GNN_official_2024`；提交 `b45546111e9dbe50e019cfa98a6c19f13a4c8aa8`。

- GraphNetwork实际依次RGCN→HypergraphConv→signed-gate highConv，与原features concat后再BiGRU。其对象是话语图，保留pregraph旁路。[forward:353](https://github.com/fufangze/SDR-GNN/blob/b45546111e9dbe50e019cfa98a6c19f13a4c8aa8/model_SDRGNN.py#L353)。
- 当前提交 `return hidden,` 返回单元素tuple；调用方 `hidden1+hidden2`是tuple拼接，分类与重构只取 `[0]`。因此speaker支路虽计算，没有进入这些输出。对HEAD源码AST和Python tuple语义已核对；核心文件与origin/main相同。[返回:389](https://github.com/fufangze/SDR-GNN/blob/b45546111e9dbe50e019cfa98a6c19f13a4c8aa8/model_SDRGNN.py#L389)、[调用:446](https://github.com/fufangze/SDR-GNN/blob/b45546111e9dbe50e019cfa98a6c19f13a4c8aa8/model_SDRGNN.py#L446)。
- 重构attention在logits之后，返回hidden也不等于分类时hidden。默认分类前向不能从它受益，训练可受重构损失影响。loss-recon开启时任务/缺失重构按 `(1-e),e` 混合，e默认0.5。[训练:368](https://github.com/fufangze/SDR-GNN/blob/b45546111e9dbe50e019cfa98a6c19f13a4c8aa8/train.py#L368)。
- 这是**当前公开检出代码的连接事实**，不是对论文作者训练版本、报告分数或全部实现的判断。它说明应当追到输出和loss，而不是默认论文仓库中每个声明模块都有实际作用。

## 对应关系的适用范围

以上代码证明几种不同的连接与训练责任：序列条件化、表示拆分、分支监督、损失侧约束、话语关系传播。它们不证明这些做法迁移到OSRAM后必然有效。OSRAM既有JEPA、InfoNCE、直接输入和短窗读取等实验，也不能因为发现参考代码使用类似机制而视为从未尝试过。
