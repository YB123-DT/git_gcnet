# GCNet Modality-JEPA Missing-Rate Sweep（seed=66）

- 状态：准备执行 `miss=0.1`、fold 1、1 epoch GPU smoke。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- 方法：训练折中心化固定 A/T/V latent；三个独立 predictor；`lambda_jepa=0.1`。
- GPU 计划：空闲时使用 GPU2–3，每卡一个进程；当前只有 GPU1 空闲，因此 smoke 临时使用 GPU1。
- 输出与 Original GCNet 完全隔离。

