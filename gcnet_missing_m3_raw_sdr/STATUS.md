# 实验状态

更新时间：2026-08-30

## 当前结论

**状态：`CLOSED — NO IMPROVEMENT`**

5 个 seed 均完成 100 epochs 和 8-rate test。Raw-Residual SDR 的 validation
8-rate mean W-F1 为 `75.5256 ± 0.5158`，低于配对 Slot-SDR 的
`77.5849 ± 0.8229`，也低于 GCNet Control 的 `78.0415 ± 1.3985`。

这否定了当前受控假设：SDR 先前的不足不能主要归因于 256D Slot 输入过早压缩。
恢复 2560D 原始模态宽度既没有带来 validation 提升，也没有改善 high-missing；
下降不是坍塌、测试解析、mask 不配对或 checkpoint 选择错误造成的。

## 完成情况

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 独立版本目录 | 完成 | `gcnet_missing_m3_raw_sdr/` |
| 向后兼容 SDR 输入 switch | 完成 | 默认 Slot 参数、state-dict 与 RNG 回归测试 |
| 七 pattern、泄漏与 backward | 完成 | Raw-SDR 模型测试 |
| 正式 5-job runner | 完成 | seeds 66–70；GPU 2/3/5/6/7 |
| 唯一 1-epoch smoke | 完成 | seed 66；validation-only；未重复 smoke |
| 正式训练 | 完成 | 5/5 seeds、500 个 epoch 记录、40 个 test NPZ |
| 独立结果审计 | 完成 | 40/40 W-F1、mask SHA、参数与 provenance 一致 |
| Checkpoint 清理 | 完成 | 5 个 `best.pt` 均已删除，SHA256 证据保留 |
| 正式判定 | FAIL | Primary gate 与 Formal gate 均失败 |

## 逐 seed 结果

| Seed | Best epoch | Raw Val-8 | Slot Val-8 | Control Val-8 | Raw Test-8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 21 | 75.8231 | 78.2020 | 79.4817 | 75.7415 |
| 67 | 24 | 75.9166 | 77.7430 | 77.1700 | 76.4421 |
| 68 | 25 | 74.7178 | 76.1528 | 77.9072 | 75.0601 |
| 69 | 34 | 75.2996 | 78.0474 | 79.3858 | 76.4466 |
| 70 | 88 | 75.8708 | 77.7794 | 76.2626 | 72.7560 |
| 均值 | — | 75.5256 | 77.5849 | 78.0415 | 75.2893 |

所有数值均为百分数。Test 只作描述，不参与模型选择或 gate。

## 预注册门槛

Primary gate 要求同时满足：

1. Raw-SDR 的 5-seed validation mean 高于 Slot-SDR；
2. 至少 `3/5` seeds 的 delta 为正；
3. high-missing (`0.4`–`0.7`) validation mean 不下降；
4. validation 无坍塌。

实际结果：

| 检查项 | 结果 | 判定 |
| --- | ---: | --- |
| Raw − Slot validation mean | -2.0593 点 | FAIL |
| 正向 seed | 0/5 | FAIL |
| Raw − Slot high-missing | -1.9933 点 | FAIL |
| Validation collapse | 0/40 rates | PASS |
| Raw − Control validation mean | -2.5159 点 | FAIL |

因此 Primary gate 为 `FAIL`，要求同时超过 Control 的 Formal gate 也为 `FAIL`。

## 审计结论

- 5 份 `history.json` 各含连续 100 epochs；
- best epoch 均由 8-rate validation mean W-F1 最大值选出；
- 40 个 NPZ 按 `labels != 0` 和预测正负号独立重算，W-F1 最大绝对误差为 `0`；
- 40 个 availability SHA 与 metrics 完全一致；
- 10 个 inherited Slot/Control reference 的 source SHA 和 realized validation/test mask
  全部通过配对检查；
- validation 和 test 共 80 个 rate 均有两类预测符号、有限输出与非零 prediction std；
- 5 个任务均 `returncode=0`，runner 为 `jobs=5, incomplete=0, failures=0`。

## 后续约束

不要再次运行同一 Raw-Residual × SDR-public 候选，也不要仅增加 seed、rate 或数据集来
延长该路线。若继续改主干，应提出与「恢复高维输入」不同的可证伪机制，并重新建立
配对 control。另一个网格搜索的 `hidden=100、window=1/1` 组合属于独立实验，不能与
这里的 `hidden=200、window=2/2` 结果混写。

完整证据见 [`results/formal/RESULTS.md`](./results/formal/RESULTS.md)。
