# One-stage Router：完成结果

INTERNAL SINGLE-SEED EXPLORATION — COMPLETE

四组训练与八率固定 mask 测试已完成。源码：
`79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`。

## 八率等权均值

| 数据集 | 模型 | 选中epoch | ACC | UA | W-F1 | 相对Flat ΔACC / ΔUA / ΔW-F1 pp |
|---|---|---:|---:|---:|---:|---|
| MELD | Flat | 37 | 60.436 | 37.504 | 57.735 | — |
| MELD | Joint Router | 37 | 60.369 | 37.431 | 57.647 | −0.067 / −0.073 / −0.088 |
| IEMOCAP-6 fold5 | Flat | 75 | 58.842 | 59.674 | 59.259 | — |
| IEMOCAP-6 fold5 | Joint Router | 94 | 59.766 | 59.635 | 59.971 | +0.924 / −0.039 / +0.712 |

## 高缺失0.5/0.6/0.7均值

| 数据集 | 模型 | ACC | UA | W-F1 |
|---|---|---:|---:|---:|
| MELD | Flat | 56.475 | 31.680 | 52.344 |
| MELD | Joint Router | 56.322 | 31.538 | 52.163 |
| IEMOCAP-6 fold5 | Flat | 57.096 | 57.787 | 57.531 |
| IEMOCAP-6 fold5 | Joint Router | 57.548 | 57.744 | 57.684 |

## 结论与直接路由贡献

MELD 未观察到正面收益。IEMOCAP-6 fold5 的 ACC/W-F1 有提升，但 UA 未提升；
不能据单 seed/fold 宣称稳定改善，也不自动扩大实验。

同一个 Joint checkpoint 内，完整候选→Router 混合的直接增量分别为：

| 数据集 | ΔACC pp | ΔUA pp | ΔW-F1 pp | 错→对 | 对→错 |
|---|---:|---:|---:|---:|---:|
| MELD | +0.0096 | +0.0066 | +0.0084 | 3 | 1 |
| IEMOCAP-6 fold5 | +0.1309 | +0.1229 | +0.1263 | 60 | 43 |

纠错/致错为八个测试条件的样本计数之和，不是互不重复的独立话语数。
IEMOCAP ACC 总增量不能全部归于推理路由：新训练后完整候选相对于配对
Flat 已提高约0.7933pp，再打开混合贡献约0.1309pp。这是描述性分解，不是
独立因果或机制证明。

逐率全部指标见 `METRICS.csv`、`SUMMARY.json` 和 `completed/`，完整自动生成
配对表见 `PAIRED_RESULT.md`。

## 原定训练配置

| 数据集 | Seed/Fold | Flat 对照 | 一阶段 Router | 主干 LR | Router LR |
|---|---|---|---|---:|---:|
| MELD | 66 / official split | 50 epochs | 50 epochs | 1e-5 | 3e-4 |
| IEMOCAP-6 | 66 / fold5 | 100 epochs | 100 epochs | 1e-3 | 3e-4 |

四组均在 biggpu GPU0/1 的持久 tmux 队列完成，每卡两组，不使用 GPU4。
原网络从随机初始化联合训练；不是旧权重解冻微调。原缺失采样、Memory
联合写入、Query、Local Skip 和分类头不变，仅增加已批准的三候选概率路由。

## 已验证

- 本地及 cmerc-repro 远程环境各61项相关测试通过；唯一警告为既有可选
  torch-sparse/libcusparse 导入警告，不用于本路径。
- 两个真实数据 GPU smoke 各3个训练 batch，主干72个参数张量与 Router12个
  参数张量均观察到有限非零梯度；原参数及 Router 实际更新，完整权重恢复成功。
- MELD / IEMOCAP-6 smoke 峰值 allocated 显存分别为1230.09 / 5382.27 MiB。
  这不是整次训练峰值或性能增益。
- 各数据集 Flat/Router 原网络初始化摘要相同、固定验证掩码摘要相同；MELD
  全50轮、IEMOCAP-6 全100轮训练 mask 摘要逐轮配对一致。
- 全32个测试预测文件 SHA256 校验通过；从原预测重算 ACC、UA、W-F1、CE，
  与保存报告差异均小于1e-10。测试话语、标签、speaker、availability 配对一致。
- 一次 Memory 扫描，额外两个分类读出使用相同 Dropout，恢复原完整前向后 RNG。
  历史 Text 来源分解是联合 Value 槽位代数，不是纯模态语义归因。

## 选点与结果边界

每个运行仅按八率验证 ACC（并列取较低 CE）选一个 checkpoint，用于全部
测试率。IEMOCAP-6 的固定训练对话 holdout 同时从新 Flat/Router 训练中排除。
这与旧逐率 Test-best 或旧冻结 miss0.3 anchor 不是同一口径，不能直接混表。
仍保留历史测试掩码/长期探索来源，不声称独立测试泛化。

队列状态为 complete，无失败训练。所有权重、优化器/RNG恢复状态、预测仍保留远程；
不上传模型或大型预测。训练/验证/测试合计耗时分别为：MELD Flat1130.37s、
Joint1884.06s；IEMOCAP-6 Flat1222.69s、Joint1823.63s。并发墙钟耗时不应相加。
队列、日志、运行 PID、环境和 smoke 完整记录见 `EXPERIMENT.json`、
`QUEUE_AT_LAUNCH.json`、`QUEUE_COMPLETE.json`、`VERIFICATION.json` 与 `smoke/`。
启动时快照保留，不回写为完成状态。远程最终产物位置：

```text
/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/
  runs/{MELD,IEMOCAPSix}/{flat,joint}/
  report/{RESULT.md,SUMMARY.json,METRICS.csv}
```
