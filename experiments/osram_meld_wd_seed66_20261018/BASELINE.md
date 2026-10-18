# 当前 MELD Flat baseline（2026-10-18）

用户已明确指定本配置为后续 MELD 对照基线。INTERNAL TEST-ORACLE SCREEN。

| 范围 | ACC (%) | UA (%) | W-F1 (%) |
|---|---:|---:|---:|
| 三种子八率平均 | 61.557 | 37.728 | 58.509 |
| 种子间样本标准差 | 0.201 | 0.620 | 0.330 |
| 高缺失率 0.5/0.6/0.7 平均 | 57.493 | 32.279 | 53.208 |

## 固定配置与汇总口径

- 模型：原 cfg84 no-JEPA Flat1600，保留 Local Skip；未加 B1/B2、Nested、B6、Gate 或其他模块。
- Adam，learning_rate=1e-5，weight_decay=1e-5，constant LR；50 epochs，batch32，dropout0.5，projector_dropout0.1。
- Local256，8 个 Memory heads，Key/Value64，causal read-before-write，write_step0.6，联合 A/T/V Block Delta Write 不变。
- MELD ECERC 官方 train/dev/test 划分；数据 SHA256：`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。
- 缺失协议固定为 `gcnet-random-mask-v1`；单模型 cyclic rates0.0–0.7，不是八个分别训练的模型。
- seeds66/67/68，各种子先等权平均八率，再等权平均三个种子。
- 每率按该次测试 ACC 选择 BEST；UA 与 W-F1 均来自该 epoch、同一次 forward、同一个 exact mask，不能重新随机测试后混填。
- 八率 BEST 可来自同一训练轨迹的不同 epoch，因此本表不是单一部署 checkpoint 的八率成绩。结果保留 Test-oracle 探索标记。
- 服务器 biggpu，环境 `/data2/yb/conda_envs/cmerc-repro/bin/python`，Torch1.13.1+cu117、NumPy1.26.4、sklearn1.3.2。

## 各种子

| Seed | ACC (%) | UA (%) | W-F1 (%) |
|---|---:|---:|---:|
| 66 | 61.786 | 38.420 | 58.890 |
| 67 | 61.470 | 37.225 | 58.324 |
| 68 | 61.413 | 37.537 | 58.312 |

## 每率三种子平均

| Missing rate | ACC (%) | UA (%) | W-F1 (%) |
|---|---:|---:|---:|
| 0.0 | 66.986 | 43.303 | 65.021 |
| 0.1 | 65.645 | 44.484 | 64.053 |
| 0.2 | 64.023 | 40.978 | 61.869 |
| 0.3 | 62.439 | 39.475 | 59.981 |
| 0.4 | 60.881 | 36.743 | 57.519 |
| 0.5 | 59.246 | 33.923 | 55.364 |
| 0.6 | 57.152 | 32.430 | 52.937 |
| 0.7 | 56.079 | 30.484 | 51.324 |

## 权重与来源保护

保留本基线 24 个 BEST checkpoint、对应 exact masks/predictions，以及各 run 的完整 `last_training.pt`。不得纳入历史调参权重清理。没有移动、覆盖、删除或重新训练任何权重。

每种子 checkpoint 目录：

- seed66：`/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1/S66_WD1`。
- seed67：`/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1/S67_WD1`。
- seed68：`/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1/S68_WD1`。

文件命名：`best_miss_0p0.pt` 至 `best_miss_0p7.pt`。完整有效配置、每率选中 epoch、checkpoint/预测/mask SHA256、原始训练 commit 与环境均在 [BASELINE.json](BASELINE.json)。登记时已重新核对三个完成状态、配置仅 seed 不同、24 个 checkpoint 与 24 个 selected-prediction 文件的 SHA256，并独立重算三种子平均。

## 历史记录边界

- 旧 lr1e-3 的 60.693/36.255/57.045 仍作为历史结果保留，其原受保护权重不删除；不再称为当前 MELD baseline。
- 旧 lr1e-5、wd2e-4 的 61.545/37.225/58.390 保留为历史配置对照。
- 现有 B1/B2 实验使用 wd2e-4，不得悄悄标成基于本 wd1e-5 baseline 的配对实验。
- 本次只登记 baseline，不修改训练默认值、不追加训练、不改变其他数据集。
