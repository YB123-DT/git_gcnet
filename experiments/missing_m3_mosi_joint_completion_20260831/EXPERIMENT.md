# CMU-MOSI Joint Completion Objective

## 问题

原 Missing-M3 Predictor 有两个彼此独立的输出：

- `reg_predictions` 接受 SmoothL1，是测试时 completion residual 真正使用的输出；
- `cl_predictions` 接受 InfoNCE，但不会被 completion residual 使用。

五种子诊断显示，前者主要输出 target-modality prototype，Real-target 与 shuffled-target
几乎不可区分。后者虽有少量样本级信息，却不能修复实际 completion 输出。

## 唯一改动

新增默认关闭的 `--jepa-contrastive-source regression`：InfoNCE 与 SmoothL1 共同监督
`reg_predictions`。默认值 `contrastive` 精确保留旧路径。没有改变：

- 模型参数与 checkpoint key；
- Natural mask、数据划分和 test mask；
- GCNet、Observed-Set Encoder、MMoE 结构和 emotion head；
- inference path；
- optimizer、epoch、learning rate 和 checkpoint selection。

## 协议

- 数据集：CMU-MOSI，fold 1；
- seeds：66--70；
- 训练：`cyclic`，一个模型轮换八个 missing rates；
- hidden 200，latent 256，slot fusion，learning rate `5e-4`；
- checkpoint：八率 validation W-F1 均值选模；
- control：直接继承
  `experiments/missing_m3_mosi_cyclic_20260830`，不重新训练；
- 主候选：`jepa_weight=0.03`；
- seed 66 额外筛选 `0.05` 与 `0.10`；
- latent 审计：test missing rate 0.7，仅作诊断，不参与选模。

## 判别顺序

1. 先检查 actual regression completion 的 centered cosine、Real-Shuffle gap、retrieval、
   effective rank；
2. latent 有样本级改善后才检查 emotion W-F1；
3. 只有五种子分类不下降才允许进入 `all` 正式协议。

## 结论

`jepa_weight=0.03` 的五种子 regression completion 确实获得更多样本级信息：

- Audio Real-Shuffle cosine gap：`0.00094 -> 0.01357`；
- Text：`0.00153 -> 0.00587`；
- Visual：`0.00155 -> 0.01168`。

但八率五种子 emotion W-F1 从 `78.0837%` 降到 `77.2784%`，差值
`-0.8053` 点；八个 rate 的均值全部为负，missing rate 0.7 为 `-1.4037` 点且
`0/5` seeds 为正。

因此本实验确认了两个独立事实：

1. 原回归补全存在 prototype shortcut，联合 InfoNCE 能直接缓解该机制失败；
2. 更真实的 latent prediction 并不会自动改善情绪识别，当前共享梯度反而与分类目标冲突。

状态：`MECHANISM_FIXED_BUT_CLASSIFICATION_FAIL`。该变体保留为机制消融，不升级为主模型，
也不扩展到昂贵的 `all` 协议或其他数据集。

