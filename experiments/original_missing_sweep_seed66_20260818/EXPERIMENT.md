# Original GCNet Missing-Rate Sweep（seed=66）

- 状态：远端完整 sweep 待 smoke gate 通过后启动。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- 远端：SSH `biggpu`，使用两张空闲 V100，每卡一个进程。
- 峰值显存：既有 miss=0.0 复现实测约 2GB。
- 输出与 Modality-JEPA 完全隔离。
- 上游 `--seed` 原本未实际初始化 RNG；本轮两个版本共同补充 Python、NumPy、Torch 与 CUDA seed，并关闭 cuDNN benchmark。这意味着先前 `miss=0.0` 复现只能作为数值链路校验，不能与本轮固定 seed 结果混为一组。
