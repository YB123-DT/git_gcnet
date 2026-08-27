# PLCI 实验结果绘图设计

## 目标

提供一个不依赖 Seaborn 的单文件绘图工具，从 PLCI 的
`fold_metrics.json` 与 Original GCNet 的 NPZ 结果中读取相同
missing rate、seed 和 dataset 的测试指标，生成可复核的数据表与论文图。

## 输入与口径

- PLCI：读取 `miss_*/seed_*/fold_metrics.json`，使用 `weighted_f1` 或
  `accuracy`。
- Original：读取 `miss_*/seed_*/**/saved/*.npz`，从所保存的测试预测重算
  指标；文件名中的两位小数只作为无法重算时的回退。
- IEMOCAP 使用多分类 weighted F1；MOSI/MOSEI 使用非零标签的正负二分类
  weighted F1，与训练器当前正式口径一致。
- 缺失项不得静默忽略；严格模式报告缺少的 method/rate/seed 组合。

## 输出

一个命令生成：

1. `scores.csv`：每个 dataset、method、rate、seed 的原始分数；
2. `mean_curve`：Original/PLCI 的均值和 sample-SD 误差带；
3. `seed_curves`：每个 seed 的完整 missing-rate 曲线；
4. `score_heatmaps`：两个方法的 seed × missing-rate 热图；
5. `delta_heatmap`：相同 seed/rate 上 PLCI−Original；
6. `delta_curve`：各 rate 的差值均值与 sample-SD；
7. PNG（300 DPI）和矢量 PDF。

图轴统一写为 `Weighted F1 (%)`，不把 weighted F1 误称为 macro F1。

## 使用边界

- 绘图脚本不参与训练，不修改结果文件。
- 配对差值只有在两边存在相同 seed/rate 时计算；当前 mask 不严格一致时，
  图注必须说明这是 seed-aligned、不是 sample-mask-paired comparison。
- 仅依赖 Python 标准库、NumPy 与 Matplotlib。

