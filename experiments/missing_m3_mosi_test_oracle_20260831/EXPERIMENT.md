# CMU-MOSI Test-Oracle 选点诊断

## 目的

检查当前 Slot Missing-M3 的 CMU-MOSI 分数是否主要受 validation checkpoint
选择影响。本实验故意使用 test set 选 epoch，存在测试泄漏，只能作为诊断，禁止进入
论文正式结果、模型选择或超参数选择。

## 锁定配置

- 数据集：CMU-MOSI；
- 模型：Slot Missing-M3，Regression-MSE，Dual-Gate Top-2/4 MMoE；
- 训练：all-rates-per-batch，100 epochs，learning rate `5e-4`；
- seed：66；
- frozen feature：wav2vec、DeBERTa、MANet；
- fixed test mask bank 与正式 seed 66 逐 rate 一致；
- 每个 epoch 在 test 的 8 个 missing rates 上评估；
- validation loader 不迭代；
- 只选择一个使 8-rate test W-F1 均值最高的 epoch，不允许逐 rate 分别选点。

## Test-Oracle 结果

最优 epoch 为 50，8-rate test W-F1 均值为 `79.427%`。

| Missing rate | W-F1 (%) |
|---:|---:|
| 0.0 | 86.408 |
| 0.1 | 84.741 |
| 0.2 | 81.747 |
| 0.3 | 80.861 |
| 0.4 | 77.852 |
| 0.5 | 77.371 |
| 0.6 | 74.030 |
| 0.7 | 72.407 |

正式 seed 66 使用 validation 选中的 epoch 43，8-rate mean 为 `79.241%`。
Test-oracle 仅增加 `+0.186` 个百分点。因此没有证据表明 MOSI 的主要差距来自
checkpoint 选坏。

注意：正式 `5e-4` 的 `78.868%` 是 seeds 66--70 的五种子均值，不能与本次单独的
seed 66 oracle 数值直接比较。

## 高缺失诊断

miss 0.7 的 686 个非零标签测试样本中，A/T/V singleton 共 543 个，占 `79.2%`：

| Pattern | 样本数 | Pattern 内 W-F1 (%) |
|---|---:|---:|
| A | 168 | 63.69 |
| T | 202 | 82.49 |
| V | 173 | 63.92 |
| AT | 38 | 94.48 |
| AV | 41 | 49.18 |
| TV | 48 | 86.88 |
| ATV | 16 | 87.30 |

0.7 下仍含 Text 的样本只有 `44.3%`。A-only 与 V-only 明显弱于 T-only，因此整体
W-F1 被 singleton pattern 构成拉低。这不是类别常量坍塌：预测同时包含正负两类，
prediction standard deviation 为 `0.982`，Accuracy 为 `72.26%`。

正式 validation-best 与 test-oracle 在 miss 0.7 分别为 `72.411%` 和 `72.407%`，
进一步说明高缺失下降不是 checkpoint 选择问题。

## 完整性审计

- 100/100 epochs 完成，远程运行耗时约 `505 s`；
- history 的 100 个 epoch 均无 `validation` 字段，均包含 8-rate `test_oracle`；
- 8/8 prediction NPZ 独立按标签正负重算 W-F1，与 metrics 一致；
- 8/8 test mask SHA256 与正式 seed 66 一致；
- 与正式 seed 66 的共享配置字段无差异；
- 日志无 traceback、OOM 或 runtime error；
- checkpoint 未同步到 Git。

## 结论

`COMPLETE — DIAGNOSTIC ONLY`。

MOSI 的主要问题不是 validation checkpoint 选择。后续若继续优化，应优先研究
A-only/V-only 表示及无 Text pattern，而不应采用 test-oracle 作为正式协议。

## 结果位置

- Remote：`/data2/yb/remote_experiments/missing_m3_mosi_test_oracle_20260831/seed_66`；
- Local：`experiments/missing_m3_mosi_test_oracle_20260831/results/seed_66`。
