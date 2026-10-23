# 第二Block能力迁移：训练实例检索审查

INTERNAL RESEARCH REVIEW ONLY — NOT IMPLEMENTED, NOT TRAINED

2026-10-23；审查起点 `d438ac5`，分支 `feature/ecerc-features`。
本轮完成实际源码与文献核对，**没有修改模型、运行推理或启动训练**。
候选新增了部署时的数据依赖，需要先确认能否接受TRAIN示例库；确认前不实施。

## 结论与范围

暂选 **TabR式动态TRAIN实例检索** 作为一个有实质区别、可以被否定的迁移候选。
不是提出了新的TabR，也不是证明它能改善OSRAM。仓库10月11日已经调研过TabR，
当时明确未授权实现/训练；这次是核对能否落实此前未实施的能力，不冒充新发现。

不同于只变换当前 `L/B/G/a`，候选每次访问其他TRAIN对话的真实示例及TRAIN标签。
原CHM仍处理当前对话的严格过去。两者信息范围不同；这也意味着不能称为同部署成本。

先看 [设计草案](DESIGN.md)。草案只建议MOSI66一个有限对照，不自动扩大种子/数据集/HPO。

## 1. 已有实验去重

| 近邻 | 实际代码 | 已有能力 | 与候选的边界 |
|---|---|---|---|
| N3 / `n3_recursive_neighbor_volumes` | `gcnet_missing_m3/meaningful_new40_retrieval.py:1–42` | 在单个已读证据集合内部找soft邻居 | 文件头明确不是其他utterance数据库 |
| C17 ProtoPNet | `gcnet_missing_m3/core20_inference.py:268–341` | TRAIN示例推送到20个类别原型参数，推理比较原型 | 不能说从未使用TRAIN示例；但不是每query检索完整TRAIN库及标签Value。其MOSI为binary CE，不是原MSE |
| B1近期历史、Teacher | 对应实验及 `osram.py` | 当前conversation严格过去的Local/模态序列访问 | 与其他TRAIN conversation的固定支持集不同；不是扩大同对话时间窗 |
| OTKE / RepSet | `meaningful_blocks_grouping.py:249`、`meaningful_v3_geometry.py:100` | 当前证据匹配可学习supports/templates | learned参数supports不是真实实例及其训练标签 |
| FactorCL / C15 | `core20_representation.py:113`、`core20.py:124–141` | 融合前真实模态shared/private/task信息目标 | 输入表征不是空白；不能再笼统推荐“先解耦再融合” |
| CED / PCIR | `coalition_mobius.py`、`model.py:472` | 当前observed集合的子集/高阶/pair交互 | 不增加跨样本证据访问 |
| TabR旧文献卡 | `docs/osram_paper_screen_20261011/SCREENING.md:19–30` | 已调研TRAIN实例检索、泄漏与缓存风险 | 写明未实现/未训练；不能把旧proposal算成负结果或新发现 |

不是已经穷尽所有数学等价实现；结论限于总账、这些实际实现与相关训练入口。

## 2. 三个同领域开源连接审查

这些例子用于识别移植条件，不用于证明“他们成功一定因为接法”。

### DLF（AAAI2025）：表示分工与中间监督不能拆开忽略

- 源码：`/data2/yb/paper/msa/repos/DLF`，commit `251bf2cb6ac3774b3c6a2159131e2bfbd4afdecf`。
- `trains/singleTask/model/DLF.py:174–206`计算specific/shared并重构，`:264–306`执行语言查询与多出口融合。
- `trains/singleTask/DLF.py:82–131`的任务监督作用于总/shared/三个specific出口，同时还有重构、正交、共享空间目标；criterion实际为L1。
- 论文Table2支持其系统中的组件/监督作用；未单独证明Norm位置或残差形式的因果优势。
- 原文与代码：[论文](https://arxiv.org/html/2412.12225v2)、[作者仓库](https://github.com/pwang322/DLF)。
- 不选作首轮：FactorCL/已有输入分工是强近邻；仅搬LFA会丢掉监督职责，语言中心设计也不能原样覆盖T缺失。

### DashFusion（TNNLS2025）：新访问来自未汇总的细粒度序列

- 源码：`/data2/yb/paper/msa/repos/DashFusion`，commit `37ba4177ba4be7db6ef2a8ae0ce916f2d6f99426`。
- `src/model/dashfusion.py:57`保留序列，`src/model/layers.py:356–379`从对齐序列形成瓶颈、更新三路原模态，并拼接四路CLS。
- 这不是“若干learned token池化后丢掉原模态”。实际跨注意力中的Norm/残差次序也不能凭类名推断。
- 论文TableIV/V的组件比较在CH-SIMS；训练还含对比项，不能归因成OSRAM下单个接法收益。
- [论文](https://arxiv.org/html/2512.05515v1)。
- 不选作首轮：当前OSRAM是每话语一个预提取向量；把Base/Gap当帧序列不提供原方法的细粒度访问能力。

### ECERC（ACL2025）：两个Text入口与不同历史集合

- 源码：`/data2/yb/paper/baseline_sources_20261006/ECERC`，commit `339fcaa68262f3fc64d860ebe6c6f79696e25818`。
- `IEMOCAP/dataloader.py:14–22`分别加载emotion/semantic RoBERTa；`train.py:80–87`作为不同输入传递。
- `model.py:262–303`构造不同speaker/因果历史集合和多条交互路径，再concat分类。
- 论文Table4有组件删除实验，不是参数匹配证明所有连接细节都优于替代。
- [论文](https://aclanthology.org/2025.acl-long.102/)。
- 不选作首轮：只重投影同一Local不能声称复制了额外semantic入口；直接历史/speaker读取又与B1近邻。

**提炼：**先核对原方法到底传递了什么，再讨论残差、维度和Norm；有些“接法”实际连同额外输入和监督一起改变。

## 3. 推荐能力的作者代码与消融依据

来源：TabR，ICLR2024。作者官方仓库已下载为小型稀疏代码副本，未安装其环境/依赖：

`/data2/yb/paper/tabr_reference_20261023.y21ZHB/tabr`

固定commit：`17baa9082506f8e7a0f8d11bb1e08212926a1507`，MIT。

- [代码](https://github.com/yandex-research/tabular-dl-tabr/blob/17baa9082506f8e7a0f8d11bb1e08212926a1507/bin/tabr.py)
- [最终会议论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/4ef594af0d9a519db8fb292452c461fa-Paper-Conference.pdf)
- `bin/tabr.py:118–133`：共享Key、label encoder、差异Value网络。
- `:219–244`：编码候选；eval明确要求目标`y is None`。
- `:253–290`：找邻居、排除self，memory-efficient模式对选中邻居重算梯度。
- `:296–307`：负平方距离softmax；Value为label embedding与key差异之和；残差接入。
- `:400–422`：候选来自TRAIN；不是eval batch中其他未知标签样本。

论文§3.2/Table2、附录A.3/Table9分别比较设计组合，不能简化成“加训练标签就会涨分”。
附录A.2/Table8分析回归label方向的修正；分类未得到同样结论。
它们均是表格任务证据，不是MOSI/MOSEI或随机缺失的已有性能证据。

## 4. 本轮交付与未做项

完成：真实代码/固定commit核对、旧方法近邻核对、最终会议版本方法与消融核对、迁移草案与泄漏边界。

未做：模型实现、参数实测、单元/GPU smoke、正式训练、任何性能结论。
没有修改OSRAM、Missing、原checkpoint、旧报告或正在运行任务；不重复提交旧实验。

仍需用户决定：是否接受推理期持有仅TRAIN构成的实例/训练标签库。
若不接受，这个候选直接排除，不暗中改造成静态原型再称为同一种能力。

实现和有限训练属于下一阶段，需对草案和预算明确确认；不是“审查完成即有效”。
