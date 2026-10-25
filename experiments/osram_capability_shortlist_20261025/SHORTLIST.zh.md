# 能力候选：融合前的模态学习竞争与互补专门化

INTERNAL RESEARCH SHORTLIST — NOT AN IMPLEMENTATION OR PERFORMANCE RESULT

源码审查锚点：`65e3a66a7da88306c28e2c0929000d0563f10b3f`，`feature/ecerc-features`。本轮只调研、去重，不修改模型、不启动训练。不把以下三个机制算成三个独立研究方向；它们属于同一个待选择的能力家族。

## 一种值得检验的能力

使 A/T/V 的可训练输入分支能够获得相对独立、互补的优化机会，而不是仅在已经生成的 Local/Base/Gap 上改变融合函数。待检验的是训练过程是否能形成更有任务价值的模态表示，不是断言 OSRAM 已被证明存在“模态竞争”。这是训练侧能力，不应为满足论文图形而伪装成新的推理 Block。

当前 `gcnet_missing_m3/model.py::ObservedSetEncoder` 使用模态独立的 `ModalityProjector`，然后融合。因此存在明确的可训练参数边界；但输入是固定预提取特征，不应宣称重新训练了原始音视频/文本大编码器。CHM 的联合写入、Query、读取和 Flat 可保持原定义。

## 三个可选择的机制，不叠加

| 机制 | 实际能力 | 来源与核查状态 | 迁移边界 |
|---|---|---|---|
| AGM，ICCV 2023 | 依据模态贡献调制训练梯度 | [论文](https://arxiv.org/abs/2308.07686)、[作者代码](https://github.com/lihongcs/AGM)；论文摘要、代码 README 已核查，未完整审计实现 | 最小侵入的训练对照；不能把较小梯度或更均衡贡献本身当成任务成功 |
| ReconBoost，ICML 2024 | 交替更新模态，并利用相对其他模态的动态目标学习互补纠错；另有巩固与联合校正 | [正式论文](https://proceedings.mlr.press/v235/hua24a.html)、[全文](https://arxiv.org/html/2405.09321v1)、[作者代码](https://github.com/huacong/ReconBoost)；实际阅读 `train_MSA.py` | 实质区别最清楚，但不是现成 OSRAM 插件；作者 MSA 脚本使用整数标签和 CE，不能原样作为当前 MOSI/MOSEI MSE 复现 |
| MLA，CVPR 2024 | 交替单模态适配，并保护共享预测映射已获得的知识 | [论文全文](https://arxiv.org/html/2311.10707v2)；阅读方法与消融；论文公布的两个 GitHub 路径本轮均返回 404 | 原方法还包括测试时不确定性融合；若只迁移训练机制，必须标明部分机制迁移，而非完整复现；代码未恢复前不优先实施 |

ReconBoost 的静态代码注意项：`train_MSA.py` 中 `ga_loss` 的两个输出都调用了 `detach()`。该行单独没有向模型传梯度，不能因为日志有此损失就声称巩固正则已生效。这是对该代码路径的检查，不是否定整篇论文或全部实现。

MLA 的消融区分共享头梯度修改与测试时融合，并非仅比较两种 Attention。其结果支持在原论文设置中独立验证优化机制，但不能外推为 OSRAM 将涨分。AGM 也不应简化为“总是削弱 Text、增强 A/V”；其贡献定义与调制规则需要忠实核查。

## 与历史实验的机制去重

- C20 CAGrad 已训练：平衡 Local/Base/Full 三个监督出口，不等同于 A/T/V 输入编码分支间的贡献调制或交替学习。旧 seed66 对照只差约 0.014 pp，不应重写为成功证据。
- FactorCL、Fishr、LUPI、RNC、双视图 InfoNCE、层级决策修正均有实际训练记录。不能把“改训练目标”“学习互补信息”泛称为从未做过。
- V3、TASR、Nested、Source-Role 与多数读出模块处理已产生的历史读值；这里改变的是输入模态分支获得任务梯度的方式。
- 在当前注册表、训练/输入库存和相关实现搜索中，未找到 OGM/AGM 或 ReconBoost 的实际实现/训练证据。这是有限范围的搜索结论，不是证明所有历史文件绝无相关尝试。

证据路径：`../osram_all_block_inventory_20261023/README.md`、`TRAINING_INPUT_NOTES.md`、`ALL_BLOCKS.md`、`RUN_RECORDS.json.gz`；`../osram_core20_20261005/RESULT.md` 与 `THREE_SEED_RESULT.md`；`../osram_r12_3seed_20261005/RESULT.md`。

## 最小检验形状，尚未授权执行

选择一种机制后再完成实现规格，不一次把 AGM、交替学习、巩固全部叠加。

建议对 ReconBoost 类互补学习采用三个对照：原 OSRAM；相同网络/预算的单纯交替更新；相同交替调度加互补目标。控制参数量、实际优化器更新次数、合法 GCNet missing 序列与训练数据暴露量；计算成本和随机数消耗另行记录，不能把相同 epoch 数等同于相同预算。原始情感任务监督始终保留，不向 Writer 输入标签，不改变联合 Block Write。回归版本的动态目标必须单独推导和测试，不能直接把分类 KL/CE 换名为 MSE。

先看固定同协议的八率及高缺失 ACC/W-F1/MSE。若互补机制不超过单纯交替且不超过原模型，就不保留该解释；若两种交替方法均改善但相互持平，只支持训练调度收益，不支持动态互补机制。任何正结果仍只是有限种子内部 Test-oracle 探索，不可直接归因为“解决模态竞争”。

## 未晋级的结构候选

真正的情绪标签空间因果状态推断，与旧 B2 的 same/change BCE、当前证据 token 上的潜状态 CRF 不完全相同，但本轮未找到足够强的现代开源迁移依据，暂不推荐启动。显式跨模态延迟估计在库存未发现实现，但当前是已对齐的 utterance 向量；没有合适粒度，不能靠 lag embedding 冒充对齐能力。

## 当前建议与限制

优先讨论 ReconBoost 类“互补专门化”是否值得迁移；AGM 可作为更低风险的比较路线，MLA 保持代码可用性待核实。尚无新结果、没有确认的缺陷、没有已完成的新 Block。如果要求必须是独立的推理期第二 Block，这份候选不满足该硬约束，应继续寻找而不是将优化器包装成网络模块。等待用户选择，不自动启动训练。
