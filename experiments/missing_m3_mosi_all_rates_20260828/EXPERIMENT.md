# CMU-MOSI All-Rates-Per-Batch

## 研究问题

MOSI train split 只有 52 段 conversation，batch 32 时每 epoch 只有 2 个 optimizer steps。原 cyclic mixed-rate trainer 每个 batch 只训练一个 rate，使每个 rate 在 100 epochs 内只获得约 25 batch、12.5 个完整语料等效曝光。本实验检验该 under-exposure 是否是 Slot Missing-M3 的主要瓶颈。

## 唯一变量

- Control（继承）：`train_rate_mode=cyclic`；
- Treatment：`train_rate_mode=all`；
- 同一个原始 batch 依次构造 8 个 rate views；
- 每个 view loss 除以 8 后 backward；
- 每个原始 batch 仍只做一次 clipping、optimizer step 和 EMA update。

模型、参数、mask schedule、loss、validation/test、checkpoint selection 全部不变。

## 协议与门槛

- CMU-MOSI official split，fold 1；
- Slot Missing-M3，hidden=200，time-attn=False，100 epochs；
- 第一阶段只运行 seed 66；
- paired Slot control：miss0 W-F1=85.69；
- 扩展门槛：miss0 W-F1 ≥87.0，nonzero-rate mean delta ≥-0.5；
- 未过门槛不扩 seeds 67–70。

## 实现验证

- TDD 红灯：缺少 `train_rate_mode`/CLI 时，focused tests 为 3 failed、1 passed；
- 实现后新增生命周期测试，验证八个 loss 的梯度平均为 `1/8`、每 batch 仅一次 optimizer/EMA；
- 代码质量审查发现 eager 构造八个 views 会同时保留大量 GPU tensors；已改为 `prepare(rate) → forward/backward(rate)` 的 lazy 顺序，并用事件顺序测试锁定；
- 本地 Missing-M3 tests：34 passed；biggpu `s0`：34 passed；
- 官方环境 1-epoch smoke：8 个 rates 各 2 batches，`optimizer_steps=2`、`ema_steps=2`；
- smoke train W-F1=0.5253、val8 W-F1=0.2633，8 个 prediction NPZ 完整；
- 规格审查与代码质量复审最终均 APPROVE。

## 判别结果

seed 66 完成 100 epochs；最佳 checkpoint 为 epoch 44，val8 W-F1=77.89%。

| Miss | All-rates W-F1 | Cyclic Slot | Delta |
|---:|---:|---:|---:|
| 0.0 | 86.56 | 85.69 | +0.87 |
| 0.1 | 84.30 | 84.30 | +0.00 |
| 0.2 | 81.34 | 80.51 | +0.84 |
| 0.3 | 81.02 | 78.11 | +2.91 |
| 0.4 | 78.59 | 76.39 | +2.21 |
| 0.5 | 75.10 | 74.02 | +1.08 |
| 0.6 | 75.11 | 73.01 | +2.10 |
| 0.7 | 72.69 | 73.20 | -0.51 |

- miss0：86.56 < 87.00，扩展门槛 FAIL；
- miss0 delta：+0.87；
- nonzero-rate mean：78.31，control 77.08，delta=+1.23；
- 7/8 rates 非负，6/8 明确为正。

因此 under-exposure 假设获得强正向证据，但未达到预注册的 miss0 扩展门槛，不运行 seeds 67–70。结果说明一个 mixed-rate 模型需要在每个 source batch 覆盖全部 rates；原 cyclic schedule 对 MOSI 的训练预算不足。但同步平均八个 rate 梯度也在后期产生过拟合：train W-F1 在 epoch 100 接近 0.93，而 val8 最佳停在 epoch 44。

## Checkpoint 诊断

- val8 最佳：epoch 44，val8=77.89，val0=84.26；
- val0 最佳：epoch 48，val0=85.20，val8=76.91；
- 当前只保存 val8-best checkpoint，因此没有用 test label 选择模型；
- 差距可能部分来自“统一八-rate鲁棒性”与“miss0 目标”之间的 checkpoint 选择冲突，但需要重新保存 val0-best checkpoint 才能验证，不能从现有 test 反推。

## 完整性与 provenance

- 8/8 prediction NPZ 的 accuracy/W-F1 独立重算一致；
- 8/8 test mask SHA256 与 cyclic Slot seed66 完全一致；
- 100 epochs 均满足每 rate 2 batches、optimizer steps=2；总计每 rate 200 batches、optimizer/EMA 200 steps；
- parameter count=32,089,733，与 Slot 完全一致；
- remote checkpoint SHA256：`a4a5f4f592596d649efbaf30d356103828372264eb1d773c887b1cd2077481d7`；
- checkpoint 留在 biggpu，轻量 config/history/metrics/prediction NPZ 已回传。
