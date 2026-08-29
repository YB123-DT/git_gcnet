# CMU-MOSI Paper-Faithful M3 MMoE

## 研究问题

当前 Single-View Missing-M3 的 `DualGateTopKMMoE` 保留了论文的 reg/cl 双 gate，
但遗漏公开代码的 task embedding、branch LayerNorm 和 residual。本实验只恢复这些
MMoE 分支机制，检验是否改善 MOSI 八个 missing rates，尤其 0.4--0.7。

## 唯一变量

- Control（继承）：`mmoe_variant=dual-gate`；
- Treatment：`mmoe_variant=paper-faithful`；
- 两者均使用独立 reg/cl gate 和共享 experts；
- Treatment 增加 reg/cl task embedding、独立 LayerNorm、GELU residual；
- Treatment 使用官方公开代码的 full-softmax-then-Top-K 权重；
- 不增加 load-balancing loss。

其余保持 CMU-MOSI、Slot、Regression-MSE、all-rates-per-batch、100 epochs、
hidden 200、window 2/2、time attention false、五个 seeds 66--70。

## 验证证据

- 远程执行唯一入口：`scripts/remote_missing_m3.sh`；
- 相关测试：77 passed；
- official 环境 1-epoch：train W-F1 0.5144，val8 W-F1 0.2544；
- 8/8 prediction NPZ、checkpoint、config/history/metrics 完整；
- regression routing entropy 1.1972，contrastive routing entropy 1.0220；
- 四个 experts 均被选择，没有初始化即 expert starvation。

## 正式任务

五个 seed 各训练一个 checkpoint、测试八个 rate。Control 直接继承：

`experiments/missing_m3_mosi_binary_task_20260829/results/SUMMARY.json`

中的 regression 五种子结果，不启动任何 Original 或 DualGate 训练。

## 状态

`COMPLETE — FAIL`。正式远程根目录：

`/data2/yb/remote_experiments/missing_m3_mosi_paper_mmoe_20260829/formal`

## 五种子结果

| Miss | Paper-faithful | DualGate control | Delta | 正向 seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 85.69 | 85.76 | -0.07 | 2/5 |
| 0.1 | 83.17 | 83.05 | +0.12 | 2/5 |
| 0.2 | 80.35 | 80.79 | -0.44 | 2/5 |
| 0.3 | 78.89 | 79.20 | -0.32 | 1/5 |
| 0.4 | 75.82 | 76.20 | -0.37 | 3/5 |
| 0.5 | 74.60 | 74.80 | -0.20 | 3/5 |
| 0.6 | 73.65 | 73.30 | +0.35 | 3/5 |
| 0.7 | 71.78 | 71.37 | +0.42 | 2/5 |

- 八-rate 均值：77.995，对照 78.059，delta=-0.064；
- 非零-rate 均值：76.897，对照 76.959，delta=-0.062；
- 0.4--0.7 均值：73.965，对照 73.916，delta=+0.049；
- 总体正向 3/5 seeds，高缺失正向 2/5 seeds；
- 40/40 treatment/control mask SHA256 配对一致；
- 新增 1,536 参数，不能形成稳定收益。

最佳 epoch 的平均 expert usage 均非零，但路由并不均衡：regression entropy 0.913，
contrastive entropy 0.744，低于四专家均匀上限 `ln(4)=1.386`。这证明统计成功捕获了
偏置，但没有证据支持在本轮追加 load-balancing loss；那将成为另一个优化变量。

## 结论

恢复 task embedding、branch norm/residual 和官方 Top-K 权重后，MOSI 八-rate 与高缺失
均值均基本不变，而且逐 seed 波动明显。因此当前 MOSI 与 CaM-HG 的差距不能归因于
MMoE 迁移不忠实。`paper-faithful` 保留为机制消融，当前 `dual-gate` 结果也不再被误称为
官方 M3 的完全复现。
