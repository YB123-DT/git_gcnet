# MELD 余弦三种子与 write_step=0.3 对照结果

INTERNAL TEST-ORACLE SCREEN

## 完成状态与协议

已全部完成。C67/C68 及 S66/S67/S68 共 5 个新增任务均为 50/50 epoch，
queue 与 RUNNER 状态为 complete、失败列表为空；C0 seed66 为已完成结果的只读复用。
没有扩展其他配置或自动启动下一轮搜索。

- 服务器：biggpu；物理 GPU5，UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`。
- 环境：`/data2/yb/conda_envs/cmerc-repro/bin/python`。
- 不可变 driver commit：`e0893a1d4a2d83f30259faab3f9aab9cdfc63eac`。
- 不可变数值源码：`b4a313841f098487a6482d2712bdac94aa8e319e`。
- 数据：受接受的 ECERC pickle，SHA256
  `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。
- MELD 官方划分、seed66/67/68、batch32、50 epoch、Adam peak LR=.001、
  weight decay=1e-5、dropout=.5；原有两轮 warmup 后 cosine 至 0。
- `gcnet-random-mask-v1`，cyclic 0.0–0.7；emotion-only/no-JEPA。
- Flat1600 与原 Local Skip 保留；8 heads、K/V64、原始联合 Block Write、
  read-before-write；没有增加第二 Block、Gate、Nested 或辅助损失。
- 两组唯一配置差异是 write_step：0.6 与 0.3。参数量均为 13,290,795。
- 主表按逐率测试 ACC 选最佳 epoch；UA/W-F1 来自选中 ACC 的同批预测。
  UA-best 是独立结果，不能与 ACC-best 混列。

本轮三组 S 与同 seed 的 cosine 对照初始化一致，训练/评测 mask 流逐 epoch
核验通过至第 50 轮。但双方选中的 best epoch 可以不同，因此不同模型的
best 预测不自动构成同一 exact mask 下的逐样本配对。

## 八率均值：ACC-best，同 epoch 对应 UA/W-F1

单位为百分比。± 为三个 seed 均值的样本标准差，不是置信区间。

| 配置 | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| cosine，step=.6 | 66 | 60.924 | 36.642 | 57.349 |
| cosine，step=.6 | 67 | 60.857 | 36.089 | 56.954 |
| cosine，step=.6 | 68 | 60.465 | 37.611 | 57.535 |
| cosine，step=.6 | mean ± SD | 60.749 ± .248 | 36.781 ± .770 | 57.279 ± .297 |
| cosine，step=.3 | 66 | 60.896 | 36.786 | 57.253 |
| cosine，step=.3 | 67 | 60.345 | 36.753 | 57.177 |
| cosine，step=.3 | 68 | 60.594 | 35.883 | 56.801 |
| cosine，step=.3 | mean ± SD | 60.611 ± .276 | 36.474 ± .512 | 57.077 ± .242 |
| .3 − .6 | mean difference (pp) | −.137 | −.307 | −.202 |

逐 seed 的 .3 − .6：

| Seed | ΔACC (pp) | ΔUA (pp) | ΔW-F1 (pp) |
|---|---:|---:|---:|
| 66 | −.029 | +.144 | −.096 |
| 67 | −.512 | +.664 | +.223 |
| 68 | +.129 | −1.728 | −.734 |

不是三种子一致收益：seed67 的 UA 提升伴随 ACC 下降，seed68 的 UA/W-F1 下降。

## 高缺失 .5/.6/.7：ACC-best

| 配置 | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| cosine，step=.6 | 66 | 56.692 | 30.401 | 51.275 |
| cosine，step=.6 | 67 | 56.667 | 29.854 | 50.947 |
| cosine，step=.6 | 68 | 56.667 | 31.544 | 52.363 |
| cosine，step=.6 | mean ± SD | 56.675 ± .015 | 30.600 ± .862 | 51.529 ± .741 |
| cosine，step=.3 | 66 | 57.037 | 29.648 | 51.342 |
| cosine，step=.3 | 67 | 56.232 | 30.186 | 51.447 |
| cosine，step=.3 | 68 | 56.552 | 31.310 | 52.098 |
| cosine，step=.3 | mean ± SD | 56.607 ± .405 | 30.381 ± .848 | 51.629 ± .409 |

## 各 missing rate：三种子均值，ACC-best

| rate | .6 ACC | .3 ACC | .6 UA | .3 UA | .6 W-F1 | .3 W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| .0 | 66.539 | 66.386 | 46.032 | 46.152 | 65.231 | 64.715 |
| .1 | 64.891 | 64.866 | 44.217 | 43.372 | 63.206 | 62.743 |
| .2 | 63.384 | 62.925 | 39.847 | 38.518 | 60.771 | 60.102 |
| .3 | 61.341 | 61.379 | 37.142 | 37.558 | 58.137 | 58.320 |
| .4 | 59.808 | 59.515 | 35.212 | 35.050 | 56.306 | 55.849 |
| .5 | 58.161 | 58.199 | 33.242 | 32.077 | 53.574 | 53.626 |
| .6 | 56.271 | 56.590 | 30.040 | 29.777 | 51.003 | 51.310 |
| .7 | 55.594 | 55.032 | 28.518 | 29.290 | 50.009 | 49.952 |

每个 seed、每个 rate 的指标、选中 epoch、七类 recall/support、模型及 mask
hash 和 artifact 路径全部保留在 `FINAL_SUMMARY.json`。

## UA-best 另列，不与主表混合

下表是按每率测试 UA 选择 epoch 的三种子八率结果，ACC/W-F1 也来自这些
UA-selected 的同次预测，不是主表 ACC-best 的对应指标。

| 配置 | ACC mean ± SD | UA mean ± SD | W-F1 mean ± SD |
|---|---:|---:|---:|
| cosine，step=.6 | 58.249 ± 1.179 | 39.808 ± .191 | 56.788 ± .505 |
| cosine，step=.3 | 58.210 ± .768 | 39.848 ± .740 | 56.655 ± .438 |

UA-best 下 .3 的 UA 均值仅 +.040 pp，同时 ACC/W-F1 略降；不能据此宣称稳定改善。

## 结论与资源

本轮结果不支持将 write_step 从 .6 改为 .3 作为默认：ACC-best 的总体 ACC、
UA、W-F1 都下降；高缺失 W-F1 小幅 +.101 pp，但 ACC/UA 没有同步改善。
保留 .6 的余弦版本作为本轮较好的训练配置，不由此宣称需要或不需要某个新模块。

用户已接受的历史 constant-LR 三种子基线是 ACC60.693/UA36.255/W-F1 57.045。
余弦 .6 相对该历史参照为 +.056/+.526/+.234 pp；C67/C68 没有本轮新跑的同 seed
constant-LR 对照，不能把这一历史参照比较说成新完成的严格配对三种子实验。

新增 C67/C68 各约 30.5 分钟，两组并行；随后 S66/67/68 各约 36.0–36.5 分钟，
三组并行。有限队列约 67.4 分钟结束，2026-10-17 15:07 UTC 完成。
这是共享资源下的实测作业时间，不是独占 GPU 的方法速度基准。

训练与选点权重、exact masks/predictions、完整恢复状态保留在原输出目录，
没有删除或覆盖历史基线。

## 输出与核验

- 远程输出：`/data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/runs_v1`。
- 复用 seed66：`/data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1/C0`。
- `FINAL_SUMMARY.json` 是远程队列结束时 `RESULTS.json` 的副本，不是重新推理。
- 该 JSON 的 SHA256：`7553e69bc8278a61d9881582e861a5b008118f92e969acba095ffe0d71a4fe37`。
- 核验 queue/RUNNER、五组 SUMMARY、各组第 50 轮 EPOCH_STATUS、选中 artifact
  存在性；核验三组 S 初始化和 mask 流通过至第 50 轮。
- 五组 ACC-selected 预测文件 hash 均匹配；UA-selected 预测的 epoch、2610
  条标签及重算 ACC/UA 与已选记录一致；每组 16 个 BEST 权重与 last state 存在。
- 六组运行、两种 selector 的八率/高缺失/三种子均值和样本 SD 均独立重算一致。
- 本次汇总没有启动训练、测试推理或修改正在运行的代码。
- 统计只作描述，不做显著性声明。所有结果均为三种子 Test-oracle 内部探索，
  不是独立测试集泛化证明。
