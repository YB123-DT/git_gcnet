# Prompt: diagnose why OSRAM contributes little on MELD

请基于下面已经确认的代码与实验事实，分析为什么因果 OSRAM/CHM 在 MELD
上的增益很小。目标是定位瓶颈，不是继续凭直觉推荐新模块。

## 模型

当前路径为：

```text
ObservedSetEncoder -> Local L_t
                  -> causal OSRAM/CHM
                     Base B_t + Gap G_t^A/G_t^T/G_t^V
                  -> Flat readout -> classifier
```

CHM 采用 read-before-write、真实观测联合 Block Delta Write，A/T/V 不是顺序
写入。Flat 同时使用 Local、Base 和 active Gap。no-JEPA、无补全、无 Teacher、
无 MMoE。MELD 使用原 GCNet batch-level missing sampler：
`gcnet-random-mask-v1`。

固定内部评价口径：每个 missing rate 分别使用 test-best checkpoint，然后对
0.0--0.7 八个 rate 求均值。不要质疑或更换这一口径，但必须将其标记为内部
Test-oracle，而不是部署成绩。

## 已确认结果

1. 原 Flat，三种子 test-best：
   - 8-rate ACC：60.498
   - 8-rate W-F1：56.754
   - high-missing ACC：56.420
   - high-missing W-F1：51.352
2. Nested GNN 三种子 test-best：
   - ACC：60.640，仅比 Flat +0.142
   - W-F1：56.828，仅比 Flat +0.074
   - high-missing W-F1 反而低 0.646
3. seed66 去掉最终 `local_skip(Local)` 并从头训练：
   - Flat：60.637 ACC / 57.087 W-F1
   - no-skip：60.560 ACC / 56.400 W-F1
   - 去掉 shortcut 没有迫使模型学到更强、更有效的历史利用。
4. 在 no-skip checkpoint 上做同 checkpoint、同 mask、单次 Memory scan 的
   冻结读出干预：
   - 去 Base：总体 W-F1 -0.618 pp，高缺失 -1.003 pp；
   - 去 Gap：总体 W-F1 -0.264 pp，高缺失 -0.491 pp；
   - Base/Gap 都去：总体 W-F1 -0.760 pp，高缺失 -1.139 pp。
   因而历史不是完全无效，但增量很小，Base 大于 Gap。
5. 屏蔽历史同时会产生纠错和致错。seed66 八率汇总：
   - 去 Base：308 次纠错、368 次致错；
   - 去 Gap：96 次纠错、121 次致错；
   - 全去：353 次纠错、433 次致错。
6. 早期 `conversation-mask-v2` 和 `gcnet-legacy-fixed-v1` 的 MELD 数字不允许
   与上述 canonical GCNet-random 结果混合。
7. IEMOCAP 上历史依赖信号明显强于 MELD，但不要在没有给出统计证据时把
   “MELD 对话更短／speaker 更多／历史更弱”直接当成已经证明的结论。

## 请回答

请把可能原因严格拆成四层：

1. **Information availability**：给定当前 Local 后，MELD 历史是否本来就几乎
   没有额外的标签信息？
2. **Storage**：历史有信息，但联合写入、保持或 speaker 混合是否使其丢失？
3. **Retrieval**：信息保存在 Memory 中，但 Base/Gap query 是否没有读到？
4. **Decision use**：Base/Gap 已含信息，但 Flat/classifier 是否没有转化为任务收益？

对每一层给出：

| 假说 | 当前证据支持什么 | 当前证据不能证明什么 | 最小可证伪实验 | 不同结果分别意味着什么 |
|---|---|---|---|---|

然后回答：现有证据最支持哪一层是主要瓶颈，置信度是多少？如果证据不足，明确
指出缺哪一项，不要强行选择。

最后只给 **一个 NEXT ACTION**。它必须是最小、能区分至少两个竞争解释的实验，
并明确：是否训练、使用哪个 checkpoint、输入/干预、指标和判定条件。

## 禁止事项

- 不要直接推荐新的 Gate、Attention、GNN、Teacher、JEPA、completion 或 auxiliary loss。
- 不要罗列二十个模块。
- 不要把“可解码”自动解释成“对原分类器有用”。
- 不要把相邻标签同/异极性称为真实 emotion shift。
- 不要使用 gold label 作为推理输入或部署规则。
- 不要把单 seed、Test-oracle 或冻结干预包装成正式论文提升。
- 不要重复已经做过的 Local/Base/Gap zeroing、去 Local shortcut 或 Nested GNN 实验。

请以证据链和可证伪性为中心，而不是以“设计一个看起来复杂的新 Block”为中心。
