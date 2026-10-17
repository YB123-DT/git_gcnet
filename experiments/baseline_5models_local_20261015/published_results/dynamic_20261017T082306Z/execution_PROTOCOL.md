# Baseline 原始动态 Mask 恢复实验

用户于2026-10-17明确指定采用保留的OSRAM Flat `gcnet-random-mask-v1`。此要求替代此前固定测试Mask要求。旧统一实现还使用了不同的独立采样近似；非零MR结果因此不能作为原协议复现。旧产物保留，未覆盖。

当前排队MELD及IEMOCAP-4/6 Session5，Seed66/67/68，MR0.0–0.7，主文五模型。360个目标配置：历史IEMOCAP ComP48项复用、MR0全模态18项复用，另294项新训练（MELD108、IEMOCAP186）。26项两Epoch GPU流程验证通过后开放对应正式任务。IEMOCAP第5折不是完整五折交叉验证；其余折和附录仍暂缓。

MELD已核查：12项MR0保留、108项新跑，已加入队列。历史ComP的60.532%来自50/25三种子，24份真实预测已重算，但不得混入300/150正式预算。300/150只找到Seed66八项分数候选，未找到完整预测/权重或Seed67/68，单独保留。

采样器逐字节复制自保留OSRAM代码。保留全局NumPy MT19937、整批比例误差循环、至少一个模态可见、32行下限、先host后guest双侧采样、padding矩形采样、OSRAM的B/T映射与padding置零。MR0.7实际缺失率为2/3。测试每批每Epoch重新采样，不声称逐样本跨模型配对。MMIN按话语Batch使用T=1。

仅恢复缺失采样及其运行记录，不改模型/损失/图/注意力/训练预算。MoMKE Stage1训练仍完整输入。MMIN兼容的完整模态Teacher直接从旧目录读取复用。

Checkpoint用Test ACC最大、同分更早Epoch；两阶段仅Stage2。ACC/UA/W-F1/Macro-F1全部来自该次选中Epoch的真实预测与Mask，立即保存。额外保存该次评估前完整RNG，验证时重放原Mask，不用新抽样的指标冒充当时最佳Epoch指标。此协议仍是test_oracle，存在乐观偏差。

运行只限本机user23的GPU0/1已记录UUID，每卡最多2任务；不碰GPU2/3，不远程训练，不自动上传。代码目录与runs/baseline_5models_dynamic_mask_20261017均与旧实验隔离。

审查入口：audit/dynamic_mask_reference.json、audit/dynamic_cpu_validation_20261017.json、IMPLEMENTATION_PLAN.md、train_one.py、dynamic_masks.py、legacy_random_mask.py。
