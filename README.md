# GCNet Completed Variants

这个仓库只展示已经完成判别实验的 GCNet 版本。四个版本共用一套数据读取、
mask bank、图构建、训练、损失和评估代码；每个版本目录只保存真正不同的模块
与锁定配置。

## 已完成版本

| 版本 | 精确替换位置 | IEMOCAPSix fixed-fold-5 结论 |
|---|---|---|
| `original` | 官方 GCNet | 8 rates × 5 seeds 的配对基准 |
| `mpfilm` | temporal/speaker 第一层 RGCN | Faithful Edge-wise 相对 Linearized 平均 `+0.000391`，不稳定，失败 |
| `cp_lecc` | temporal/speaker 第一层 RGCN | corrected protocol 只完成 `.5/.7`，整体晋级失败 |
| `sequence_aff` | `hidden1 + hidden2` 分支融合 | `0.623101` vs Original `0.626974`，失败 |

这些结果使用固定的 IEMOCAP-6 第 5 fold，不是五折交叉验证均值。

## 目录

```text
common/gcnet/                    # 四版本共享 GCNet 主干
versions/original/              # 官方 no-op 配置
versions/mpfilm/                # Missing-Pattern FiLM RGCN
versions/cp_lecc/               # Complete-Preserving Low-Rank ECC
versions/sequence_aff/          # mask-conditioned Sequence AFF
results/iemocap6/fold5/         # 紧凑结果与 provenance
environment/                    # 唯一共享环境
provenance/source_map.json      # 新路径到历史 commit 的映射
run.py                          # 唯一运行入口
```

## 统一运行入口

```bash
python run.py --version original --help
python run.py --version mpfilm --help
python run.py --version cp_lecc --help
python run.py --version sequence_aff --help
```

正式训练参数仍通过共享 GCNet CLI 提供。例如：

```bash
python run.py \
  --version original \
  --dataset IEMOCAPSix \
  --fold-index 5 \
  --seed 66 \
  --mask-seed 66 \
  --mask-type constant-0.5 \
  --data-root /path/to/IEMOCAP \
  --mask-bank-root /path/to/mask_banks \
  --output-dir /path/to/output \
  --base-model LSTM \
  --loss-recon
```

`--graph-conv-variant` 和 `--branch-fusion` 由版本配置锁定，不能在命令行
覆盖，防止版本名称与实际运行结构不一致。

## 数据与大型实验资产

Git 只保存代码、Markdown、JSON 和小型汇总。以下内容不进入仓库：

- wav2vec/DeBERTa/MANet 特征；
- 原始数据集与完整 feature archives；
- mask banks；
- checkpoint、原始 NPZ 和逐 epoch 日志。

每个结果目录的 `provenance.json` 记录原始服务器 artifact root、source
commit、fold、seeds 和实际完成的 missing rates。

## 范围边界

本发布入口不包含尚未完成的候选。研究历史仍保留在本地 Git 分支中，但只有
通过锁定实验并完成结果审计的方法才会加入 `versions/` 和 `results/`。
