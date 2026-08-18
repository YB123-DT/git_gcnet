# Original GCNet Missing-Rate Sweep（seed=66）

- 状态：完成。8 个 missing rate 均为 5 folds × 100 epochs，seed=66。
- 范围：IEMOCAP-Six，missing rate 0.0–0.7，五折，100 epochs。
- 远端：SSH `biggpu`，使用两张空闲 V100，每卡一个进程。
- 峰值显存：既有 miss=0.0 复现实测约 2GB。
- 输出与 Modality-JEPA 完全隔离。
- 上游 `--seed` 原本未实际初始化 RNG；本轮两个版本共同补充 Python、NumPy、Torch 与 CUDA seed，并关闭 cuDNN benchmark。这意味着先前 `miss=0.0` 复现只能作为数值链路校验，不能与本轮固定 seed 结果混为一组。

## 最终 Weighted-F1

| Missing rate | Mean | Population std |
|---:|---:|---:|
| 0.0 | 0.61729 | 0.02730 |
| 0.1 | 0.60843 | 0.02618 |
| 0.2 | 0.59845 | 0.02747 |
| 0.3 | 0.59123 | 0.03626 |
| 0.4 | 0.57303 | 0.05139 |
| 0.5 | 0.56221 | 0.03460 |
| 0.6 | 0.55854 | 0.04401 |
| 0.7 | 0.57040 | 0.04269 |

0.6 首次运行遇到远端 CUDA driver 初始化失败；CUDA 健康检查通过后在同一配置下补跑成功。Original 单实验 `nvidia-smi` 峰值约 2.2–2.4GB。
