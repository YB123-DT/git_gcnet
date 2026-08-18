# GCNet Modality-JEPA Missing-Rate Sweep（seed=66）

- 状态：第一轮 `miss=0.1`、fold 1、1 epoch GPU smoke 已通过；最佳 epoch RNG 重放 smoke 运行中。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- 方法：训练折中心化固定 A/T/V latent；三个独立 predictor；`lambda_jepa=0.1`。
- 远端：SSH `biggpu`；正式实验使用两张空闲 V100，每卡一个进程。
- 输出与 Original GCNet 完全隔离。
- 第一轮 smoke：35,966,654 参数；比原始 34,140,166 新增 1,826,488（5.35%）；PyTorch peak allocated 1353.09MB；训练/验证/测试及诊断均为有限值。
- 1 epoch 仅验证链路，不解释分类效果；Real–Shuffle gap 为 Audio 0.0024、Text 0.0008、Visual 0.0049。
