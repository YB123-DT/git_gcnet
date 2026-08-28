# MOSI Modality-Slot Fusion 设计规格

## 目标

在不修改 Missing-M3 的训练目标、EMA teacher、六方向 MMoE predictor、GCNet 图主干和混合 missing-rate 协议的前提下，修复当前 observed-set 等权平均对 CMU-MOSI 强 Text 信号的稀释问题。

本轮只有一个自变量：`ObservedSetEncoder` 的融合方式。

## 现有问题

当前节点表示为：

\[
u_i=\rho\left(\frac{1}{|O_i|}\sum_{m\in O_i}(s_i^m+e_m)+e_{p_i}\right).
\]

该表示在进入融合层前已经把三个模态压到同一向量。融合层无法恢复每个维度来自 Audio、Text 还是 Visual，也无法针对 MOSI 学习 Text 主导、Audio/Visual 条件修正的组合。

当前混合率模型在 CMU-MOSI 的八个测试 missing rates 上得到 85.44、82.52、80.72、79.06、76.84、74.06、74.27、71.29 W-F1；完整率也比既有完整模态 M3 结果低约 1.18 点，因此优先检查输入集合融合，而不是扩大 predictor。

## 候选方案与决定

### A. 保留等权均值

参数少且满足集合置换不变性，但已知会在融合前丢失模态槽位身份。本轮不采用，保留为现有对照。

### B. Modality-Slot Fusion（采用）

为 A/T/V 保留固定槽位：

\[
v_i^m=a_i^m(s_i^m+e_m),
\]

\[
u_i=\rho_{\mathrm{slot}}([v_i^A;v_i^T;v_i^V;e_{p_i}]).
\]

缺失槽位严格为零；pattern embedding 使用独立的第四槽位。`rho_slot` 将 `4d` 映射回原来的 `d`，因此 GCNet 接口不变。固定槽位使模型能够学习模态非对称性，同时没有引入样本级 reliability scalar 或 attention。

### C. 模态注意力融合

能够动态分配权重，但同时引入权重归一化、注意力参数和解释变量，无法判断提升来自模态身份保留还是注意力选择。本轮拒绝，只有 Slot Fusion 失败后才重新讨论。

## 代码接口

`ObservedSetEncoder` 新增：

```python
fusion_type: str = "mean"
```

允许：

- `mean`：保持现有公式、参数形状和 checkpoint 行为；
- `slot`：采用四槽位拼接融合。

`MissingM3GraphModel`、`TrainConfig` 和 CLI 逐层传递该选项：

```text
--fusion-type mean|slot
```

默认值必须为 `mean`，以免改变已经完成的 IEMOCAP/MOSI/MOSEI 结果及旧命令。

## 保持不变

- 三个 modality-specific Student Projector；
- modality embedding 与七种 availability pattern 编码；
- EMA Teacher 及其更新顺序；
- M3 六方向 Top-K MMoE predictor；
- JEPA loss、temperature 与 loss weight；
- Temporal/Speaker GCNet、分类或回归头；
- 一个 mixed-rate checkpoint 测试八个 missing rates；
- dataset split、seed、mask schedule、优化器和 epoch selection。

本轮明确不研究六方向梯度冲突，也不运行 `lambda_J=0`、attention 或参数匹配控制。

## 正确性测试

1. `mean` 模式在同一 RNG、参数和输入下保持现有输出及 state-dict 结构。
2. `slot` 支持 A、T、V、AT、AV、TV、ATV 七种 pattern。
3. 修改缺失模态的输入数值不能改变 `slot` 输出，证明无 missing-value leakage。
4. A/T/V 槽位顺序固定；同一个 projected latent 放入不同槽位时产生不同融合输入和可学习输出。
5. padding 节点仍输出全零。
6. CLI 与 checkpoint config 记录 `fusion_type=slot`。
7. CPU 单元测试与远程 GPU 前向、反向均为有限值。

## 正式实验

- Dataset：CMU-MOSI；
- Seeds：66、67、68、69、70；
- Training：每个 seed 训练一个 mixed-rate model；
- Test：同一最佳 checkpoint 测试 0.0–0.7 八个 rates；
- Existing `mean` 结果直接继承，不重新训练；
- GPU：0、1、2、3、5，禁用 GPU 4；
- 结果必须保存每个 rate 的 W-F1、MAE、correlation、mask SHA256、最佳 epoch 和参数量。

## 解释与失败判定

若 Slot Fusion 在多数 rates 上提高 W-F1，尤其改善 0.0–0.4，则证据支持“等权集合均值稀释 MOSI 的模态非对称信息”。若仅个别 seed 提升或整体下降，则不能继续归因到 predictor 冲突；应将 Slot Fusion 判为失败并恢复 `mean`，再单独设计下一项实验。

