# 从已有最强改版继续：选择依据与单项修改提议

INTERNAL DIAGNOSTIC ONLY

2026-10-03，分析基于仓库 c7ecd8b。仅核对已完成实验和当前实现；
没有修改模型、启动训练、新增推理或改变 checkpoint。
这是待讨论的修改提议，不是已经批准／实现／验证的实验。

## 选择口径

沿用用户此前偏好的从头一阶段联合训练。以 MOSI seeds 66/67/68、
100 epochs、完整八率 W-F1 算术均值为主，高缺失 .5/.6/.7 为辅。
不以单 seed 最高分、最后一轮或事后标签路由成绩替代此口径。
下面审计的是近期 cfg84 no-JEPA 读出／训练改版，不宣称覆盖仓库所有历史模型。
所有分数都是 per-rate BEST Test-oracle；排序本身也利用了测试结果，
仅供内部探索，不能据此作正式选模或泛化优势声明。

## 已完成三种子候选

W-F1 (%)；各 seed 内先平均 rates，再等权平均 seeds。

| 版本 | 8-rate | High | 结果来源（experiments/ 下） |
|---|---:|---:|---|
| 原 cfg84 Flat（参考） | 80.559 | 75.594 | osram_decision_correction_20261003/results/baseline |
| Relation + 双读出监督 | 80.315 | 75.366 | osram_relation_dual_readout_20261003/MULTISEED.md |
| Local Evidence Gate + L2 | 80.286 | 75.955 | osram_local_evidence_gate_20260930/RESULT.md |
| 双视图 task-only rho=.10 | 80.260 | 75.297 | osram_paired_history_rho025_20261002/THREE_SEED_RESULT.md |
| Evidence Gate L1 | 80.203 | 75.153 | osram_evidence_gate_l1_20260930/RESULT.md |
| Local Skip Gate | 80.193 | 75.331 | 下述远程原始结果 |
| Hierarchical Evidence Gate | 80.018 | 75.267 | osram_hierarchical_evidence_gate_20260930/RESULT.md |
| 双视图 task-only rho=.25 | 80.014 | 75.274 | osram_paired_history_rho025_20261002/THREE_SEED_RESULT.md |
| Feature-only Gate | 79.957 | 75.158 | osram_feature_only_gate_20260930/RESULT.md |
| Memory Shift D3-W256 | 79.953 | 74.992 | osram_shift_capacity_20261002/THREE_SEED_RESULT.md |
| 原 Memory Shift D1-W128 | 79.949 | 75.206 | 下述远程原始结果 |
| 双视图 task-only rho=.05 | 79.927 | 75.315 | osram_paired_history_rho025_20261002/THREE_SEED_RESULT.md |
| History-input Gate | 79.892 | 75.239 | osram_cfg84_history_input_gate_20260929/RESULT.md |
| Random-only history-query adapter | 79.824 | 74.649 | 下述远程原始结果 |
| Memory-only Adapter | 79.679 | 74.600 | osram_memory_only_adapter_20261001/RESULT.md |
| Gap-increment Filter | 79.579 | 74.699 | osram_gap_increment_filter_20261003/RESULT.md |
| Post-GRN | 79.521 | 74.665 | osram_cfg84_post_grn_20260929/RESULT.md |
| Decision Correction | 79.290 | 74.164 | osram_decision_correction_20261003/RESULT.md |

结果分析技能要求区分两个排序：Relation dual 是八率的数值第一，L2 Gate
是本表高缺失的第一。Relation 与 L2 Gate 八率仅差 0.0297 pp，不能
宣称前者显著更好。Relation 三个 seed 八率均低于原 Flat。
若坚持整体指标优先，则选择 Relation dual 作为修改起点，原 Flat 仍是基线。

冻结 Flat 的第二阶段 Evidence Gate best 为 80.741/75.782，但 best
包含原 Flat epoch0（24 个 seed-rate 项中 5 项选中 epoch0），而且是
额外训练预算／冻结主干协议，不纳入一阶段排名。History-Innovation
有未完成 seed，不把残缺运行混作完整三种子候选。单 seed Relation、
四格 splice、readout intervention 也不作为三种子训练模型排名。

## 只提议一项修改：Relation 条件输入 stop-gradient

保留选中版本的原 Flat、Relation 结构与 0.5/0.5 task loss：

```text
u = LocalSkip(L) + Adapter([L, Base, masked Gap])
delta = Relation(stop_gradient(L), stop_gradient(Base),
                 stop_gradient(masked Gap), availability, umask)
pred_base = H(LN(u))
pred_full = H(LN(u + delta))
loss = 0.5 task(pred_base, y) + 0.5 task(pred_full, y)
```

只截断 Relation 分支从其输入返回 Encoder/Memory 的直接梯度；不 detach
原 Flat 输入、u、delta 或 full 输出，不冻结任何原模块。Relation 的
Local/history 投影、MLP 和输出层继续训练；原主干仍通过 Flat 路径接受
两项任务损失监督。一阶段、一个 missing mask、一条 Memory 轨迹。
推理公式与参数量不变（仍 13,789,441），不加 Gate、loss、Teacher
或第二次 Memory query，也不改 loss 权重、宽度、dropout、学习率。
这里指在选中版本代码上做受控改动，不是加载其 checkpoint 进行第二阶段训练。

理由来自旧单损失 Relation 的已完成 residual-off 诊断：seed66 八率
R-off - Flat = -0.707，而 R-on - R-off = +0.216。它提供了“分支本身
与主路径训练变化可能作用不同”的动机，但不是双读出三种子的机制证据。
新提议只检验：去掉额外的 Relation-to-upstream 梯度路径是否有益。

关键限制：Full loss 仍通过 LN(u+delta) 和 Flat 返回主干，delta 的数值
仍影响该梯度，因此这不保证消除主干漂移，更不保证 W-F1 提升。
当前没有测得梯度冲突，不能声称已证实冲突或其根因。

未选的方向：继续加深 Relation/Gate 会再改容量；调整双读出 loss 比例
会改目标。此次只提出梯度路径这一项，不同时做以上两项或超参搜索。

## 若后续实施，必须验证的边界

- 默认关闭新选项，原行为／checkpoint／RNG 不变。
- 相同权重、输入及 RNG 下，打开 detach 前向输出完全相同。
- 单独对 Relation residual 求导：新版本不回传到 L/Base/Gap，但
  Relation 参数有有限梯度；检查时须让零初始化输出层已经非零。
- 完整双读出 loss 下，Encoder、Memory、Flat、head 仍有任务梯度；
  不是借 detach 实现冻结或第二阶段训练。
- 原首句、inactive Gap、padding 安全屏蔽规则保持。
- 后续效果仍看全测试集八率及高缺失，不以 near/opposite 单格定成功；
  若复评四格，继续固定原 Flat Local-only 分组。

这些是验收建议，本次没有实现或运行上述测试；设计技能使本轮停留于
候选和方案讨论，不自动开训。

## 本次核验记录

已从十个本地完整变体的三个 metrics/config 重算数值，核对八率完整、
100 epochs 配置、seed、cyclic 模式、Test-oracle 协议，以及与原 Flat
逐 seed 的八个 evaluation mask hashes 相同。其余表项沿用已完成
多种子报告。Post-GRN 保留原报告中的 CUDA exact-equality 历史 caveat，
不把它称为完整等价验证通过。

另外只读 SSH 查询 biggpu 以下三个 runs/seed_{66,67,68} 的 metrics、
history 和 PROVENANCE，均为 100 条 history、status=complete、八率
Test-oracle；此处未重新审计这三个远程变体的全部源码／权重：

- `/data1/yb/remote_experiments/osram_local_skip_gate_20261001/runs`
- `/data2/yb/remote_experiments/osram_cfg84_memory_shift_residual_20260928/runs`
- `/data2/yb/remote_experiments/osram_cfg84_history_query_random_20260928/runs`

每 seed 的 8-rate/high 读值依次为：

| 远程版本 | seed66 | seed67 | seed68 |
|---|---|---|---|
| Local Skip | 80.420183 / 75.749346 | 80.261142 / 75.888366 | 79.897707 / 74.356160 |
| Memory Shift D1 | 80.496759 / 75.812634 | 80.396085 / 76.092373 | 78.955137 / 73.714220 |
| History-query random | 79.908887 / 74.865451 | 79.875633 / 74.986855 | 79.687088 / 74.094317 |

实现核对：`gcnet_missing_m3/osram.py::CurrentHistoryRelationBlock` 与
Relation 分支接入，`train_gcnet.py::_relation_dual_readout_loss`。
现有调用未 detach Relation 输入；diagnostics 的 detach 不是该修改。
