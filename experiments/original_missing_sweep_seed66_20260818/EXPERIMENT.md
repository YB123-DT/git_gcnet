# Original GCNet Missing-Rate Sweep（seed=66）

- 状态：等待 GPU；`miss=0.0` 已有独立复现结果可校验。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- GPU 计划：空闲时使用 GPU0–1，每卡一个进程。
- 峰值显存：既有 miss=0.0 复现实测约 2GB。
- 输出与 Modality-JEPA 完全隔离。

