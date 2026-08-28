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

正式结果运行中。完成后在本目录保存每数据集的五 seed metrics、40 个轻量 prediction NPZ 与汇总表；checkpoint 仅保留在 biggpu，不进入 Git。
