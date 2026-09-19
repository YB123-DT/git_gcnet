# OSRAM 实验版本矩阵

本文件把当前 OSRAM 代码、实验入口和历史研究分开记录。数据、checkpoint、远程日志和临时诊断输出不提交到 Git；它们只保存在远程实验目录中。

## 当前可复用基线

当前统一诊断基线是 **cfg84 no-JEPA**：

- causal OSRAM（`osram_bidirectional=false`）；
- `osram_write_step=0.6`；
- Flat/mean readout；
- `osram_output_dim=1600`，8 heads，key/value dimension 64/64；
- `training_objective=emotion-only`，禁用未使用的 EMA Teacher/MMoE；
- cyclic mixed-rate training；
- 每个 seed、每个 missing rate 独立用 Test W-F1 选 epoch（`per-rate-test-oracle`）。

核心实现仍集中在：

```text
gcnet_missing_m3/model.py
gcnet_missing_m3/osram.py
gcnet_missing_m3/train_gcnet.py
gcnet_missing_m3/loss.py
```

## 实验入口

| 用途 | 入口 | 说明 |
|---|---|---|
| MOSI no-JEPA 基线 | `experiments/osram_no_aux_cfg84_20260919/run.py` | 五 seed，当前 cfg84 基线 |
| IEMOCAP-4/6 五 session | `experiments/osram_iemocap_cfg84_nojepa_5session_20260919/run.py` | S1–S5 leave-one-session-out，seed 66/67/68 |
| MOSEI no-JEPA | `experiments/osram_mosei_cfg84_nojepa_20260919/run.py` | 官方 split，seed 66/67/68 |
| reg-only 对照 | `experiments/osram_reg_only_20260919/run_cfg84_capacity.py` | 仅保留 JEPA regression 分支，不作为 no-JEPA 控制 |

IEMOCAP 与 MOSEI 入口只保存配置和运行逻辑；输出位置由脚本中的远程 `OUTPUT_ROOT` 指定，不把训练结果混入源码目录。

## 研究分支边界

以下分支是独立研究路线，不要把它们混入 cfg84 no-JEPA 基线：

| 分支 | 内容 | 是否为当前基线 |
|---|---|---|
| `feature/osram-complete` | Complete-State / Text-subspace 诊断 | 否 |
| `feature/osram-reg-only` | MMoE regression-only 研究 | 否 |
| `feature/osram-pam-text` | PAM-T 缺失 Text 记忆 | 否 |
| `feature/osram-pam-episodic-text` | PAM-E 诊断 | 否 |
| `feature/text-core` | Text-Core task space | 否 |
| `feature/osram-uniform-forced-text` | 当前 cfg84/训练协议与实验入口 | 是（本版本） |

## 运行协议

IEMOCAP 五折使用 S1–S5 留一 session 测试；MOSEI 使用官方 train/validation/test split。两者当前诊断都使用：

```text
cyclic mixed-rate training
8 missing rates: 0.0 ... 0.7
per-rate Test-oracle checkpoint selection
```

这是内部诊断协议，不应直接写成论文正式结果。正式结果必须切换到 validation 选点并锁定同一比较协议。

## 不提交内容

- `dataset/**/.gcnet_cache/`
- 本地 `experiments/**/results/` 与临时 smoke 输出
- 远程 `/data2/yb/remote_experiments/` 下的 checkpoint、history、日志
- 任意数据集特征和模型权重
