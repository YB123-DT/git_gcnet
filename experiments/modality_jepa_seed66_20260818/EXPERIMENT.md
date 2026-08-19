# GCNet Modality-JEPA Missing-Rate Sweep（seed=66）

- 状态：完成。8 个 missing rate 均为 5 folds × 100 epochs，seed=66。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- 方法：训练折中心化固定 A/T/V latent；三个独立 predictor；`lambda_jepa=0.1`。
- 远端：SSH `biggpu`；正式实验使用两张空闲 V100，每卡一个进程。
- 输出与 Original GCNet 完全隔离。
- 第一轮 smoke：35,966,654 参数；比原始 34,140,166 新增 1,826,488（5.35%）；PyTorch peak allocated 1353.09MB；训练/验证/测试及诊断均为有限值。
- 1 epoch 仅验证链路，不解释分类效果；Real–Shuffle gap 为 Audio 0.0024、Text 0.0008、Visual 0.0049。
- **后续 parity 更正：** `missing_rate=0` 时 JEPA objective 与 Predictor 均严格关闭；独立运行得到的 -1.43 points 是 GCNet/PyG CUDA 训练轨迹噪声，不能作为 JEPA 效果。证据见 `../miss0_parity_20260819/EXPERIMENT.md`。后续 miss=0 直接复用同一 baseline checkpoint 和指标。

## 最终对照

| Missing rate | Original W-F1 | JEPA W-F1 | JEPA − Original | 正增益 folds |
|---:|---:|---:|---:|---:|
| 0.0 | 0.61729 | 0.60296 | -0.01433 | 1/5 |
| 0.1 | 0.60843 | 0.61940 | +0.01096 | 3/5 |
| 0.2 | 0.59845 | 0.60834 | +0.00989 | 3/5 |
| 0.3 | 0.59123 | 0.56919 | -0.02205 | 3/5 |
| 0.4 | 0.57303 | 0.59381 | +0.02078 | 4/5 |
| 0.5 | 0.56221 | 0.56388 | +0.00167 | 1/5 |
| 0.6 | 0.55854 | 0.54591 | -0.01263 | 2/5 |
| 0.7 | 0.57040 | 0.55446 | -0.01595 | 1/5 |

最稳定的正向点是 missing rate 0.4（+2.08 points，4/5 folds）；0.1 和 0.2 为约 +1 point，但只有 3/5 folds 为正；0.5 的均值增益很小且只有 1/5 folds 为正，不能视为稳定改善。高缺失率 0.6、0.7 均下降，因此不支持“缺失越高收益越大”。

所有非零 missing rate 的三模态平均 Real–Shuffle cosine gap 均为正：Audio 约 0.30–0.36、Text 约 0.096–0.112、Visual 约 0.16–0.21。预测任务确实使用了上下文，而非仅输出与随机 target 同等的常量；但可学习的预测任务并不保证 ERC 一致提升。

运行异常：GPU4 两次将 JEPA 0.1 进程以 SIGKILL 终止，随后永久避开 GPU4，在 GPU0 补跑成功。四个 JEPA 同时运行时因每进程默认 40 个 Torch CPU threads 导致 160 线程争抢 80 核；通过暂停两路、两两执行并自动恢复保留了已有训练状态。JEPA `nvidia-smi` 峰值约 2.3–2.7GB。

结果汇总：`../missing_sweep_seed66_summary.json`。
