# Missing-M3 Raw-Residual SDR 诊断

本目录检验一个单变量假设：此前 SDR 只接收 256D Slot 表示，效果不佳是否主要由
`2560D → 256D` 的图前压缩造成。

结论已经闭合：**不是。** 在保持数据、mask、优化、Missing-M3 Predictor 和验证选择
规则不变时，让 SDR 直接接收 2560D Raw-Residual 输入，5-seed validation 8-rate
W-F1 反而比 Slot-SDR 低 `2.0593` 个百分点。因此该路线标记为
`CLOSED — NO IMPROVEMENT`，不再重复扩展 seed 或数据集。

## 唯一处理变量

Slot-SDR 对照：

```text
official incomplete A/T/V features
  → Student Projectors
  → Slot Observed-Set Encoder [256D]
  → SDR-public
```

Raw-SDR 候选：

```text
official incomplete A/T/V features [512 + 1024 + 1024]
  → modality-specific Student Projectors [256D]
  → zero-initialized residual adapters [512 / 1024 / 1024]
  → observed raw block + student residual
  → concatenate [2560D]
  → SDR-public
  → MOSI regression head

same student latents + SDR hidden + availability
  → existing Missing-M3 Predictor
  → EMA target and JEPA loss during training only
```

对模态 `m`，输入写成：

```text
student_m = Projector_m(feature_m)
adapted_m = availability_m × (feature_m + Adapter_m(student_m))
```

Adapter 最后一层以零初始化，因此训练开始时，候选就是严格的 observed raw input。
缺失模态块与 padding 始终保持精确零值；Missing-M3 Predictor、EMA Teacher 和
JEPA loss 没有改变。

## 锁定协议

- 数据集：CMU-MOSI；
- 特征：冻结的 wav2vec、DeBERTa、MANet utterance-level 特征；
- seeds：66、67、68、69、70；
- 每个 seed 训练 1 个 `all-rates-per-batch` 模型；
- 每个最佳 validation checkpoint 测试 `0.0` 至 `0.7` 共 8 个 missing rates；
- checkpoint 按 validation 8-rate mean W-F1 选择，test 不参与选择；
- 配置：`hidden=200`、`graph_hidden=100`、`window_past=2`、
  `window_future=2`、`time_attention=False`、`lr=5e-4`、100 epochs；
- GPU：seed 66/67/68/69/70 分别使用 2/3/5/6/7，明确排除 GPU 4；
- Slot-SDR 和 GCNet Control 全部继承，未重新训练。

注意：另一个 MOSI 网格搜索得到的 validation-selected 最佳组合是
`hidden=100、window=1/1、time_attention=False、lr=5e-4`。本诊断没有套用该组合，
因为 Raw-SDR 必须与既有 Slot-SDR 和 Control 保持逐 seed、逐 mask、逐配置配对；若同时
改变窗口或 hidden，就无法归因于「恢复高维输入」这一变量。

## 正式结果

| 模型 | Validation 8-rate W-F1 | Validation high-missing W-F1 | Validation miss-0 W-F1 |
| --- | ---: | ---: | ---: |
| Raw-Residual SDR | 75.5256 ± 0.5158 | 71.3728 ± 1.4140 | 83.0868 ± 2.2034 |
| Slot-SDR | 77.5849 ± 0.8229 | 73.3661 ± 1.2111 | 85.1941 ± 1.2324 |
| GCNet Control | 78.0415 ± 1.3985 | 74.3224 ± 1.9821 | 84.9271 ± 1.8378 |

Raw-SDR 相对 Slot-SDR：

- mean delta：`-2.0593` 个百分点；
- 正向 seed：`0/5`；
- high-missing delta：`-1.9933` 个百分点；
- validation 与 test 的 40 个 rate 均未出现单符号或常量输出坍塌。

完整逐 seed、逐 rate 数字、统计分析与独立审计见
[`results/formal/RESULTS.md`](./results/formal/RESULTS.md)。

## 参数与运行画像

| 版本 | 注册参数 | 可训练参数 | Backbone 参数 |
| --- | ---: | ---: | ---: |
| Raw-Residual SDR | 15,643,426 | 14,783,266 | 12,209,701 |
| Slot-SDR | 12,486,434 | 11,626,274 | 9,444,901 |

Raw 输入令总参数增加 3,156,992。五个正式任务平均耗时约 291.44 秒，最大峰值显存为
601,360,384 bytes。该运行画像只描述当前 V100 环境，不能当作跨模型速度结论。

## 运行方式

检查固定的 5-job 矩阵：

```bash
python -m gcnet_missing_m3_raw_sdr.run_mosi \
  --source-commit <40-character-git-sha> \
  --dry-run
```

正式运行：

```bash
python -m gcnet_missing_m3_raw_sdr.run_mosi \
  --source-commit <40-character-git-sha>
```

Runner 会在任何训练进程启动前审计 10 个 inherited references，并锁定配置、源码、
环境、参数量、validation/test mask 哈希和结果 schema。正式完成后只保留 checkpoint
SHA256，不保留 `best.pt`。

## 文件说明

- [`model.py`](./model.py)：锁定 Raw-Residual + SDR-public 的薄模型；
- [`train_gcnet.py`](./train_gcnet.py)：只暴露 lifecycle 参数的正式训练入口；
- [`run_mosi.py`](./run_mosi.py)：5-job 调度、配对审计、恢复和 validation-only gate；
- [`tests/`](./tests)：输入、泄漏、梯度、训练器和 runner 合同测试；
- [`STATUS.md`](./STATUS.md)：当前状态与关闭依据；
- [`results/formal/`](./results/formal)：小型正式结果与 provenance，不含 checkpoint、
  特征或预测 NPZ。

设计与执行约束见：

- [`2026-08-30-missing-m3-raw-residual-sdr-design.md`](../docs/superpowers/specs/2026-08-30-missing-m3-raw-residual-sdr-design.md)
- [`2026-08-30-missing-m3-raw-residual-sdr.md`](../docs/superpowers/plans/2026-08-30-missing-m3-raw-residual-sdr.md)
