# One-stage Router：启动与验证记录

INTERNAL SINGLE-SEED EXPLORATION — TRAINING IN PROGRESS

源码：`79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`。本页不是最终性能报告。

| 数据集 | Seed/Fold | Flat 对照 | 一阶段 Router | 主干 LR | Router LR |
|---|---|---|---|---:|---:|
| MELD | 66 / official split | 50 epochs | 50 epochs | 1e-5 | 3e-4 |
| IEMOCAP-6 | 66 / fold5 | 100 epochs | 100 epochs | 1e-3 | 3e-4 |

四组均已在 biggpu GPU0/1 的持久 tmux 队列启动，每卡两组，不使用 GPU4。
原网络从随机初始化联合训练；不是旧权重解冻微调。原缺失采样、Memory
联合写入、Query、Local Skip 和分类头不变，仅增加已批准的三候选概率路由。

## 已验证

- 本地及 cmerc-repro 远程环境各61项相关测试通过；唯一警告为既有可选
  torch-sparse/libcusparse 导入警告，不用于本路径。
- 两个真实数据 GPU smoke 各3个训练 batch，主干72个参数张量与 Router12个
  参数张量均观察到有限非零梯度；原参数及 Router 实际更新，完整权重恢复成功。
- MELD / IEMOCAP-6 smoke 峰值 allocated 显存分别为1230.09 / 5382.27 MiB。
  这不是整次训练峰值或性能增益。
- 各数据集 Flat/Router 原网络初始化摘要相同、固定验证掩码摘要相同；启动后
  已完成的 MELD 首1轮、IEMOCAP-6 首3轮训练 mask 摘要分别配对一致。
- 一次 Memory 扫描，额外两个分类读出使用相同 Dropout，恢复原完整前向后 RNG。
  历史 Text 来源分解是联合 Value 槽位代数，不是纯模态语义归因。

## 选点与结果边界

每个运行仅按八率验证 ACC（并列取较低 CE）选一个 checkpoint，用于全部
测试率。IEMOCAP-6 的固定训练对话 holdout 同时从新 Flat/Router 训练中排除。
这与旧逐率 Test-best 或旧冻结 miss0.3 anchor 不是同一口径，不能直接混表。
仍保留历史测试掩码/长期探索来源，不声称独立测试泛化。

训练尚未完成，不填写最终 ACC/UA/W-F1。后台队列将在四组结束后自动校验
初始化、训练/验证/测试 mask 配对并生成完整报告；失败运行不会参与结果排名。
队列、日志、运行 PID、环境和 smoke 完整记录见 `EXPERIMENT.json`、
`QUEUE_AT_LAUNCH.json` 与 `smoke/`。远程最终产物位置：

```text
/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/
  runs/{MELD,IEMOCAPSix}/{flat,joint}/
  report/{RESULT.md,SUMMARY.json,METRICS.csv}
```
