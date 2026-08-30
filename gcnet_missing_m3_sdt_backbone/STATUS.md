# 实验状态

更新时间：2026-08-30

## 当前结论

**状态：`PENDING — FORMAL RESULTS NOT RUN`**

等活跃参数 SDT-style 主干、锁定训练入口、5-seed runner 和真实 batch 资源画像已经放在
本目录。正式 5-seed × 8-rate 训练结果尚未完成，因此当前没有可报告的 treatment
W-F1，也不能判断该主干是否优于 GCNet control。

`PROFILE.json` 中的 loss、速度和显存仅来自 1 个真实训练 batch，用于确认代码路径与
`jobs_per_gpu=2` 的资源决策，不能当作 validation/test 结果。

## 进度

| 项目 | 状态 | 证据或后续动作 |
| --- | --- | --- |
| 独立版本目录 | 完成 | `gcnet_missing_m3_sdt_backbone/` |
| 等活跃参数主干 | 完成 | Active 5,869,370；control 5,864,700 |
| Missing-M3 合同接入 | 完成 | 保留 Observed-Set、Predictor、EMA 和 loss |
| 模型、训练器、runner 测试 | 待最终回归 | `tests/` 已提供合同测试；提交前运行完整目标测试 |
| 真实 batch profiling | 完成 | `PROFILE.json`；V100 32GB；candidate 0.5352 GiB |
| 并发计划 | 锁定 | 2 jobs/GPU；只使用 GPU 0、1、2 |
| Seeds 66–70 正式训练 | `PENDING` | 固定源码提交后，运行 runner 并传入 `--source-commit <40-character-git-sha>` |
| 五种子结果汇总 | `PENDING` | 正式任务完成后生成结果摘要 |
| GitHub 结果归档 | `PENDING` | 只归档验证后的源码与小型结果文件 |

## 锁定运行矩阵

| Seed | GPU | Treatment | Original/control |
| ---: | ---: | --- | --- |
| 66 | 0 | 待运行 | 继承，不重跑 |
| 67 | 1 | 待运行 | 继承，不重跑 |
| 68 | 2 | 待运行 | 继承，不重跑 |
| 69 | 0 | 待运行 | 继承，不重跑 |
| 70 | 1 | 待运行 | 继承，不重跑 |

每张 GPU 最多同时运行 2 个 job。所有 seed 使用相同的冻结 wav2vec、DeBERTa、MANet
特征和 `all-rates-per-batch` 协议。

## 继承的严格对照

| Validation 指标 | Control |
| --- | ---: |
| 8-rate mean W-F1 | 78.7675 |
| High-missing (`0.4`–`0.7`) mean W-F1 | 74.9589 |
| Miss-0 W-F1 | 85.6461 |

这些数值来自已有 strict control，只用于配对比较。本 runner 不构造、不训练 Original。

## 验收门槛

候选必须同时满足：

1. 5-seed validation 8-rate mean ≥ 79.2675；
2. 至少 4/5 seeds 的 validation 8-rate delta > 0；
3. high-missing validation mean ≥ 74.9589；
4. miss-0 validation ≥ 85.3461；
5. 无单一符号、常量输出、非有限 loss 或表示异常坍塌。

若未满足，状态改为 `CLOSED — NO IMPROVEMENT`，结论仅限于：在当前冻结特征与锁定
协议下，整体替换 GCNet 主干不足以通过预注册门槛。不能据此声称 Transformer 或图网络
在一般情况下孰优孰劣。

## 归档约束

- 不提交 `best.pt`、`predictions_miss_*.npz`、数据集或大型特征；
- 不把 profiling loss 写入正式结果表；
- 正式结果必须能追溯到 seed、配置、100-epoch history、mask SHA 和日志；
- manifest 必须记录 source commit、源码 SHA、环境、feature 路径、配置 SHA 与参数口径；
- test 结果完整保存，但不用于回头修改本候选。
