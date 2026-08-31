# Joint Completion 结果摘要

## Emotion W-F1

| Missing rate | Joint 0.03 | Cyclic control | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 83.9900 | 84.7383 | -0.7483 | 2/5 |
| 0.1 | 81.8056 | 82.6343 | -0.8287 | 1/5 |
| 0.2 | 79.9664 | 80.5326 | -0.5662 | 1/5 |
| 0.3 | 78.3447 | 79.0067 | -0.6619 | 2/5 |
| 0.4 | 75.5165 | 76.2348 | -0.7183 | 1/5 |
| 0.5 | 73.9061 | 74.9427 | -1.0366 | 0/5 |
| 0.6 | 73.7960 | 74.2749 | -0.4789 | 1/5 |
| 0.7 | 70.9018 | 72.3055 | -1.4037 | 0/5 |
| Mean | **77.2784** | **78.0837** | **-0.8053** | — |

五种子 mean delta：seed 66 `+0.0054`、67 `-1.5606`、68 `+0.1886`、
69 `-0.8243`、70 `-1.8356`。seed 66 的中性结果不能代表五种子结论。

## Regression completion，miss 0.7

| Target | Metric | Control | Joint 0.03 |
|---|---|---:|---:|
| Audio | Centered cosine | 0.02194 | 0.03779 |
| Audio | Real-Shuffle cosine | 0.00094 | 0.01357 |
| Audio | Retrieval / chance | 1.8x | 3.0x |
| Text | Centered cosine | 0.01766 | 0.02503 |
| Text | Real-Shuffle cosine | 0.00153 | 0.00587 |
| Text | Retrieval / chance | 1.6x | 3.0x |
| Visual | Centered cosine | 0.02277 | 0.02661 |
| Visual | Real-Shuffle cosine | 0.00155 | 0.01168 |
| Visual | Retrieval / chance | 0.8x | 1.0x |

联合目标显著削弱了 prototype shortcut，但 Visual retrieval 仍在随机机会附近，而且 emotion
分类稳定下降。因此结果是机制诊断成功、候选方法失败。

## Seed 66 权重筛选

| JEPA weight | 8-rate W-F1 | 相对 control | 判断 |
|---:|---:|---:|---|
| 0.03 | 79.0188 | +0.0054 | 进入五种子，最终失败 |
| 0.05 | 78.8242 | -0.1893 | 不扩展 |
| 0.10 | 78.3899 | -0.6236 | 不扩展 |

