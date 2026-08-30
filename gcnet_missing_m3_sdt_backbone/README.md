# MOSI 等活跃参数对话主干诊断

本目录用于回答一个受控问题：在冻结特征、缺失协议、Observed-Set 编码、Missing-M3
预测器和训练目标不变时，用全上下文 Transformer 整体替换 GCNet 对话主干，能否改善
CMU-MOSI 的多缺失率表现。

这是 **whole-backbone diagnostic（整体主干诊断）**，不是新的 Transformer 或 SDT
方法。SDT、CSS 等工作已采用全上下文注意力、位置编码和 speaker embedding；本实现只将
这一类已知结构作为对照，用于定位 GCNet 对话主干是否构成当前瓶颈。

## 受控变量

保持不变：

- 冻结的 wav2vec、DeBERTa、MANet utterance-level 特征；
- Slot Observed-Set Encoder 与显式 availability 输入；
- Missing-M3 Predictor、EMA Teacher 及其损失；
- `all-rates-per-batch` 训练协议与 `0.0` 至 `0.7` 的 8 个 missing rates；
- 数据划分、mask schedule、评估指标和 checkpoint 选择规则；
- `hidden=100`、`window_past=1`、`window_future=1`、`lr=5e-4`、100 epochs。

唯一处理变量：

```text
Slot observed-set node [L, B, 256]
  → Linear(256, 384)
  → sinusoidal position encoding
  → speaker embedding
  → 5 × full-context Pre-LN Transformer
       8 heads, FFN 384 → 704 → 384
  → LayerNorm
  → Linear(384, 250)
  → ReLU
  → classification hidden [L, B, 250]
```

Attention 只屏蔽 padding，不使用 causal mask。Missing-M3 Predictor、EMA Teacher 和
分类头仍接收与 control 相同语义的输出，不新增 completion、蒸馏、LoRA、CNN 或图模块。

## 参数预算

参数比较采用实际进入前向的 active/effective 口径：

| 主干 | 注册参数 | Active/effective 参数 | 相对 control |
| --- | ---: | ---: | ---: |
| GCNet control | — | 5,864,700 | — |
| SDT-style candidate | 5,869,754 | 5,869,370 | +4,670（+0.07963%） |

Candidate 的注册参数包含 384 维 padding speaker row；该行在设计上不参与有效位置计算，
因此从 effective 参数中扣除。Control 原类中另有不会进入锁定配置前向的 GRU 与
MatchingAttention 参数，不能把它们计入 active budget。参数近似相等不表示 FLOPs
相等，因此资源画像单独记录在 [`PROFILE.json`](./PROFILE.json)。

## 实验协议

- 数据集：CMU-MOSI；
- seeds：66、67、68、69、70；
- GPU 映射：66→0、67→1、68→2、69→0、70→1；
- 每个 seed 训练 1 个 mixed-rate 模型，并在最佳 validation checkpoint 上评估全部
  8 个 missing rates；
- 最佳 epoch 按 validation 8-rate mean W-F1 选择；并列时保留最早 epoch；
- Existing Original/control 结果直接继承，runner 不包含 Original 训练命令。

## 真实批次资源画像

`PROFILE.json` 使用 PyTorch 1.8.0 和 Tesla V100-SXM2-32GB，对同一个真实 CMU-MOSI
训练 batch 执行全部 8 个 rate 的前向、反向，以及 1 次 optimizer/EMA 更新。它不是
1-epoch smoke，也不产生模型效果结论。

| 模型 | 单真实 batch（8 rates） | 峰值已分配显存 | 注册参数 | 可训练参数 |
| --- | ---: | ---: | ---: | ---: |
| SDT-style candidate | 1.1368 s | 0.5352 GiB | 8,847,037 | 7,986,877 |
| Missing-M3 GCNet control | 1.8875 s | 0.4063 GiB | 11,121,383 | 10,261,223 |

Candidate 的实测峰值显存远低于 32 GiB。正式 runner 因此锁定
`jobs_per_gpu=2`，在 GPU 0、1、2 上一轮容纳 5 个 seed。该决定只依据容量画像，不代表
正式训练耗时或最终分数。

## 解释边界

- CMU-MOSI 在当前数据接口中只有 1 个 speaker，有效位置的 speaker embedding 是常量
  条件。因此本实验不能支持「学习了 speaker interaction」这一结论。
- Control 的 legacy recurrent 路径会处理 padding；candidate 使用显式 padding mask。
  这是整体主干替换留下的语义差异。已有 Packed-control 未形成稳定提升，本轮不重复训练，
  但汇总结论必须披露该混杂。
- 本实验只覆盖当前冻结特征与锁定 mixed-rate 协议，不能外推为 Transformer 与图网络的
  一般优劣比较。

## 运行

先检查将要启动的 5 个 treatment 命令：

```bash
python -m gcnet_missing_m3_sdt_backbone.run_mosi \
  --jobs-per-gpu 2 \
  --source-commit <40-character-git-sha> \
  --dry-run
```

正式运行：

```bash
python -m gcnet_missing_m3_sdt_backbone.run_mosi \
  --jobs-per-gpu 2 \
  --source-commit <40-character-git-sha>
```

runner 只接受固定的 GPU 集合 `0 1 2`，支持基于完整输出的恢复运行。单个 seed 只有在
`config.json`、100-epoch `history.json`、含完整 8-rate test 的 `metrics.json` 和
`train.log` 都通过检查时才视为完成；validation-only 不属于正式完成结果。重跑 pending
任务前会清除旧完成文件，避免新 config 与旧 history/metrics 拼接。Manifest 额外绑定
source commit、源码 SHA、完整配置 SHA、feature 路径、运行环境、mask SHA 和参数口径；
不能仅凭旧 manifest 或进程退出码跳过任务。

## 结果判定

正式结果只按预注册 validation 门槛判断，不根据 test 反向改模型：

- 5-seed validation 8-rate mean 至少达到 `79.2675`，即比 strict control
  `78.7675` 高 `0.50` 个百分点；
- 至少 4/5 seeds 的 validation 8-rate delta 为正；
- high-missing（`0.4`–`0.7`）validation mean 不低于 control `74.9589`；
- miss-0 validation 不低于 control `85.6461 - 0.30 = 85.3461`；
- 不出现单一符号输出、常量输出、非有限 loss 或表示异常坍塌。

当前正式分数为 `PENDING`。详见 [`STATUS.md`](./STATUS.md) 和
[`results/README.md`](./results/README.md)。

## 文件说明

- [`model.py`](./model.py)：Pre-LN Transformer 主干及 Missing-M3 集成；
- [`train_gcnet.py`](./train_gcnet.py)：锁定的 CMU-MOSI 训练入口；
- [`run_mosi.py`](./run_mosi.py)：5-seed 并发、恢复和 manifest 管理；
- [`PROFILE.json`](./PROFILE.json)：真实 batch 资源画像；
- [`tests/`](./tests)：模型、训练器和 runner 合同测试；
- [`results/`](./results)：正式结果与归档规则。

完整设计与实现约束见：

- [`docs/superpowers/specs/2026-08-30-mosi-sdt-backbone-diagnostic-design.md`](../docs/superpowers/specs/2026-08-30-mosi-sdt-backbone-diagnostic-design.md)
- [`docs/superpowers/plans/2026-08-30-mosi-sdt-backbone-diagnostic.md`](../docs/superpowers/plans/2026-08-30-mosi-sdt-backbone-diagnostic.md)
