# Single-View Missing-M3 多数据集扩展

## 目标

在不修改 IEMOCAPSix 已验证机制的前提下，将一个 mixed-rate checkpoint 测试八个 missing rates 的协议扩展到：

- IEMOCAPFour：fold 5；
- CMU-MOSI：官方 train/validation/test split；
- CMU-MOSEI：官方 train/validation/test split。

三个数据集均运行 seeds `66, 67, 68, 69, 70`。Original 结果继承，不重新训练。

## 不变项

- `Observed-Set Fusion → GCNet Temporal/Speaker backbone`；
- 训练期 M3 Top-2/4-expert MMoE；
- EMA teacher，tau 0.996；
- `lambda_J=0.1`；
- batch 均衡轮换 `0.0–0.7`；
- 八 rate validation W-F1 等权平均选一个 checkpoint；
- 同一 checkpoint 测八个固定 test masks；
- 无 reconstruction，无第二 view，无测试时 completion。

## 数据集适配

| Dataset | Fold | Task loss | Checkpoint/Test metric | Speakers | Output |
|---|---:|---|---|---:|---:|
| IEMOCAPFour | 5 | Cross-entropy | Weighted F1 | 2 | 4 classes |
| CMU-MOSI | 1 | MSE | Nonzero-label Acc-2 / W-F1 | 1 | scalar |
| CMU-MOSEI | 1 | MSE | Nonzero-label Acc-2 / W-F1 | 1 | scalar |

MOSI/MOSEI 同时记录 MAE 与 Pearson correlation，但 checkpoint 仍按八 rate W-F1 平均选择，以保持三个数据集生命周期一致。

## 真实 batch 验证

| Dataset | Feature dims | Classification/Regression loss | JEPA loss | Missing targets | Peak GPU |
|---|---|---:|---:|---:|---:|
| IEMOCAPFour | 512/1024/1024 | 1.3911 | 3.8205 | 1719 | 971.66 MiB |
| CMU-MOSI | 512/1024/1024 | 2.3211 | 3.6327 | 1195 | 652.96 MiB |
| CMU-MOSEI | 512/1024/1024 | 1.1389 | 2.9481 | 308 | 488.71 MiB |

全部使用 `eta=0.5`，完成 forward、backward、optimizer step 与 EMA update。

## 结果状态

15/15 个正式任务完成。每个数据集保存五 seed metrics、40 个轻量 prediction NPZ 与汇总表；checkpoint 仅保留在 biggpu，不进入 Git。

## IEMOCAPFour 结果

| Miss | S66 | S67 | S68 | S69 | S70 | W-F1 Mean ± SD |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.48 | 84.63 | 86.45 | 84.45 | 85.29 | 85.26 ± 0.79 |
| 0.1 | 84.85 | 85.32 | 86.50 | 84.36 | 85.31 | 85.27 ± 0.79 |
| 0.2 | 84.81 | 83.49 | 86.43 | 83.31 | 84.32 | 84.47 ± 1.25 |
| 0.3 | 83.78 | 84.18 | 84.74 | 83.76 | 83.71 | 84.03 ± 0.44 |
| 0.4 | 83.65 | 82.59 | 84.98 | 82.26 | 83.73 | 83.44 ± 1.07 |
| 0.5 | 80.78 | 81.49 | 83.88 | 81.68 | 84.45 | 82.46 ± 1.61 |
| 0.6 | 80.78 | 82.77 | 81.84 | 80.79 | 83.23 | 81.88 ± 1.12 |
| 0.7 | 82.51 | 82.87 | 81.08 | 79.08 | 82.48 | 81.60 ± 1.57 |

最佳 epoch：S66=60、S67=69、S68=75、S69=94、S70=90。

## CMU-MOSI 结果

| Miss | S66 | S67 | S68 | S69 | S70 | W-F1 Mean ± SD | Acc-2 | MAE | Corr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 86.18 | 83.60 | 87.27 | 85.68 | 84.45 | 85.44 ± 1.44 | 85.43 | 0.862 | 0.767 |
| 0.1 | 84.05 | 80.14 | 85.35 | 81.59 | 81.45 | 82.52 ± 2.12 | 82.47 | 0.909 | 0.726 |
| 0.2 | 82.39 | 79.98 | 81.98 | 80.55 | 78.71 | 80.72 ± 1.50 | 80.70 | 0.949 | 0.691 |
| 0.3 | 80.25 | 75.87 | 81.64 | 81.21 | 76.35 | 79.06 ± 2.75 | 79.02 | 0.979 | 0.666 |
| 0.4 | 79.13 | 72.72 | 80.17 | 77.03 | 75.14 | 76.84 ± 3.01 | 76.83 | 1.027 | 0.620 |
| 0.5 | 74.11 | 72.87 | 78.83 | 74.08 | 70.44 | 74.06 ± 3.05 | 74.09 | 1.061 | 0.590 |
| 0.6 | 73.57 | 74.77 | 76.23 | 75.00 | 71.80 | 74.27 ± 1.67 | 74.30 | 1.060 | 0.585 |
| 0.7 | 72.61 | 72.58 | 70.13 | 73.61 | 67.51 | 71.29 ± 2.47 | 71.37 | 1.109 | 0.539 |

最佳 epoch：S66=44、S67=92、S68=41、S69=50、S70=79。

## CMU-MOSEI 结果

| Miss | S66 | S67 | S68 | S69 | S70 | W-F1 Mean ± SD | Acc-2 | MAE | Corr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 86.87 | 85.85 | 87.34 | 85.19 | 86.88 | 86.43 ± 0.88 | 86.48 | 0.527 | 0.787 |
| 0.1 | 85.99 | 85.48 | 86.38 | 84.97 | 85.73 | 85.71 ± 0.53 | 85.78 | 0.543 | 0.769 |
| 0.2 | 85.50 | 84.68 | 84.65 | 83.35 | 85.12 | 84.66 ± 0.81 | 84.75 | 0.560 | 0.751 |
| 0.3 | 84.85 | 83.58 | 84.31 | 82.69 | 84.47 | 83.98 ± 0.86 | 84.09 | 0.575 | 0.736 |
| 0.4 | 83.86 | 83.15 | 82.97 | 82.40 | 83.31 | 83.14 ± 0.53 | 83.30 | 0.589 | 0.721 |
| 0.5 | 82.99 | 82.35 | 82.16 | 81.31 | 82.62 | 82.28 ± 0.63 | 82.44 | 0.603 | 0.703 |
| 0.6 | 81.60 | 81.90 | 81.97 | 80.48 | 81.15 | 81.42 ± 0.62 | 81.62 | 0.618 | 0.687 |
| 0.7 | 80.68 | 81.60 | 80.65 | 78.67 | 81.63 | 80.65 ± 1.20 | 80.88 | 0.633 | 0.665 |

最佳 epoch：S66=16、S67=32、S68=27、S69=42、S70=30。

## 完整性检查

- IEMOCAPFour：1241 条 test utterances/rate，40/40 NPZ 指标重算一致，四类均被预测。
- CMU-MOSI：686 条 test utterances/rate，40/40 NPZ 的 nonzero-label Acc-2/W-F1、MAE 重算一致。
- CMU-MOSEI：4659 条 test utterances/rate，40/40 NPZ 的 nonzero-label Acc-2/W-F1、MAE 重算一致。
- 每个 seed 具有 100 条 epoch history、一个 best epoch、八个不同 test mask SHA256。
- 无 NaN、无进程失败、无分类单类坍塌。

## 解释

- 一个 checkpoint 跨八 rates 在 IEMOCAPFour 与 MOSEI 上表现稳定；MOSEI 的五 seed 方差尤其小。
- MOSI 的 high-missing 下降更明显，且 seed 方差更大，说明小数据集上的 mixed-pattern multi-task optimization 更敏感。
- 当前实验同时改变 observed-set node construction、mixed-rate training 和 M3 objective；在运行相同模型但 `lambda_J=0` 的控制前，不能把全部收益归因于 M3。
- 不同 source-target tasks 在共享 Projector/GCNet/experts 上仍可能产生梯度冲突；当前结果只能排除灾难性冲突，不能证明梯度协同。
