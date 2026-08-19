# GCNet / Modality-JEPA Session-5 曲线异常诊断

日期：2026-08-19

## 结论

旧表不能作为 CaM-HG / SDR-GNN 协议结果。异常曲线不是汇总脚本或标签数量
错误，而是单次训练轨迹的巨大方差，加上 test-set checkpoint selection、随机
test mask 和协议不对齐共同造成的。

## 直接复现证据

所有复现均使用 IEMOCAP-Six Session 5、1,623 utterances、seed 66 和原实验
参数，只使用 GPU0，未使用 GPU4。

| Condition | 原结果 W-F1 | 同 seed 重跑 W-F1 | 差值 |
|---|---:|---:|---:|
| JEPA, missing=0.3 | 46.68 | 61.81 | +15.13 |
| GCNet path, missing=0.7 | 64.25 | 59.49 | -4.76 |

JEPA 0.3 原运行在 epoch 100 的 train W-F1 只有 57.05；重跑达到 80.73。
原运行的 predictor effective rank 为 A/T/V = 3.22/4.89/2.03，重跑恢复为
7.22/12.61/5.53。因此 46.68 是一次优化轨迹坍塌，不是 missing=0.3 的稳定
效果。

GCNet 0.7 的 64.25 已从保存的 1,623 条 logits 独立重算，指标本身没有读错；
但同 seed 重跑只得到 59.49，证明 64.25 是不可复现的单次高点。

## 已排除原因

1. 不是 test mask 的实际缺失率异常：missing=0.3 的 Baseline 和 JEPA 保存
   mask 均为 30.46% missing，三个模态分布平衡。
2. 不是结果汇总读错：从保存 logits 重新 argmax 后仍复现旧 W-F1。
3. 不是 JEPA loss 在 miss=0 偷跑：独立 parity 已证明 loss 和 predictor gradient
   均为零；两个 Baseline CUDA 副本也会在 Adam step 后快速分叉。

## 根因

### 1. 把单次 trial 当成 SDR-GNN 报告值

SDR-GNN 论文 4.2 节明确说明测试结果取 10 trials 平均。旧实验每个 rate 只有
seed 66 的一次训练，因此 CUDA/PyG 图聚合和 Adam 放大的训练方差没有被平均。

### 2. 使用 test session 选择 checkpoint

IEMOCAP loader 返回 `val_loader=test_loader`。每个 epoch 先在 Session 5 上计算
validation W-F1，再选择该 epoch，并在同一 Session 5 上报告 test W-F1。这是
test leakage；随机 mask 又使 validation/test 两次评估不是同一输入，进一步放大
checkpoint 选择噪声。该行为继承自参考代码，但不应继续用于新实验。

### 3. 每次评估重新生成 mask

`random_mask` 在每个 validation/test call 内重新采样。旧结果同时包含模型训练
随机性和 test corruption 随机性，没有预生成固定 test masks，无法做严格 paired
comparison。

### 4. 实际并未复现 SDR-GNN 超参数协议

旧命令固定使用 `LSTM, hidden=200, window=2, lr=0.001`，且总损失为
`CE + Recon (+ 0.1 JEPA)`。SDR-GNN 官方示例使用 `GRU, hidden=200,
window=3`，其 reconstruction 配比为 `0.5*CE + 0.5*Recon`；论文还对每个
missing rate 搜索 `hidden in {100,150,200,250}` 与 `window in {1,2,3,4}`。
因此“直接取五折 fold 5”只对齐了 test session，没有对齐训练协议。

### 5. missing=0.7 实际是 2/3

为保证每条 utterance 至少保留一个模态，三模态 `random_mask` 在请求 0.7 时
进入 one-hot 分支，每条恰好保留一个模态，实际 missing rate 为 66.67%。论文将
0.7 描述为对 2/3 的近似；不能把它解释为严格 70%。

### 6. CaM-HG 与 SDR-GNN 的训练 mask 也不完全相同

SDR-GNN 对每个 missing rate 使用固定 rate 训练；CaM-HG 自己的模型在训练时
按 batch 从 `Uniform(0,1)` 动态采样 missing rate，并另外使用 text blindness。
CaM-HG 表中的旧 baseline 数字来自 SDR-GNN，但 CaM-HG 本身不是同一训练策略。

## 后续有效协议

1. 固定 Session 5 为 test，从 Session 1--4 按 conversation 划独立 validation。
2. 每个 trial 和 missing rate 预生成 train/validation/test mask；Baseline 与 JEPA
   共享相同 mask 文件。
3. test mask 在 checkpoint selection 期间不可读取；只在最终模型确定后运行。
4. 至少 10 个独立 training trials，报告 mean and standard deviation；不能只对
   同一 checkpoint test 10 次。
5. miss=0 复用同一 Baseline checkpoint；非零 missing rate 做 paired seeds。
6. 分开报告两套协议：SDR-style fixed-rate training 与 CaM-HG-style dynamic-rate
   training，不混为一张主表。

机器可读复现结果：

- `jepa_miss03_rerun_fold_metrics.json`
- `gcnet_miss07_rerun_fold_metrics.json`
