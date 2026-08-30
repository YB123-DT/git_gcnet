# CMU-MOSI Slot Missing-M3 固定缺失率实验

## 状态

`RUNNING`。

本实验检验现有统一八率训练是否限制了 MOSI 表现；它只改变训练/选模/测试协议，不修改
Slot Missing-M3 模型或损失。

## 研究问题

现有正式模型在每个 batch 同时学习八个 missing rates，并用八率 validation W-F1 均值
选择一个 checkpoint。SDR-GNN 等结果常采用一个 rate 训练一个模型。为了区分“模型能力不足”
和“统一训练的负迁移”，本实验注册：

```text
train eta = validation-selection eta = test eta
```

## 锁定矩阵

- 数据集：CMU-MOSI，官方 split，fold 1；
- rates：`0.0, 0.1, ..., 0.7`；
- seeds：`66, 67, 68, 69, 70`；
- 总任务：`8 × 5 = 40`；
- frozen features：wav2vec-large-c、DeBERTa-large-4、MANet；
- Slot observed-set + Original GCNet path；
- DualGate MMoE、EMA teacher、target-balanced JEPA；
- regression MSE，validation W-F1 选模；
- hidden 200，latent 256，window 2/2，time attention off；
- Adam，LR `5e-4`，weight decay `1e-5`，JEPA weight `0.1`；
- 100 epochs，batch size 32；
- GPU 0/1/2，每卡最多 5 个并发任务；GPU 4 禁用。

Original 和既有 mixed-rate Slot Missing-M3 结果直接继承，不重新训练。

## 完整性要求

每个完成目录必须同时具备：

- 与注册 rate/seed 一致的 `config.json`；
- 恰好 100 epochs 的 `history.json`；
- 只用同一个 rate 选模且只测试该 rate 的 `metrics.json`；
- 只含该 rate 的 prediction NPZ；
- test mask SHA256；
- `train.log` 和原子 `status.json`。

完成 40/40 后，从 NPZ 独立重算 W-F1，再与既有 mixed-rate 五种子均值按 rate 比较。

## 路径

- Remote formal：`/data2/yb/remote_experiments/missing_m3_mosi_fixed_rate_20260830/formal`；
- Runner：`scripts/run_mosi_fixed_rate.py`；
- 设计：`docs/superpowers/specs/2026-08-30-mosi-fixed-rate-protocol-design.md`。

checkpoint 和 prediction NPZ 仅留在 biggpu；Git 只归档小型结果、摘要和 provenance。
