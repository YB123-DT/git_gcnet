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

`RUNNING`。正式远程根目录：

`/data2/yb/remote_experiments/missing_m3_mosi_paper_mmoe_20260829/formal`

