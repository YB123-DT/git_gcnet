# OSRAM causal η=0.6 predictor 复查

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

## 结论

**没有证据表明 η=0.6 重训练修复了缺失 Text latent 的样本辨识能力。**
在自然 miss=0.5 的 A/V/AV 三个子群中，regression head 的 Text centered cosine、
Real−Shuffle cosine gap、retrieval 五种子均值均低于 η=1。不能把 raw cosine 约 0.93
解读为正确预测了具体样本。下降并非每个 seed 都一致，也未做显著性声明。

本次只运行已训练 checkpoint 的 evaluation，未修改模型、loss、写入规则或训练任何回灌模型。

## 范围与可复现口径

- CMU-MOSI，seeds 66–70；重点沿用此前退化的 **natural miss=0.5**，不是覆盖全部 rates。
- A、V、AV 是同一正式 mask 下自然出现的 utterance 子群，不重新遮挡整个对话。
  因此目标 utterance 无 Text，但其对话上下文可以有其他 utterance 的真实 Text。
- η=.6 checkpoints：`/data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_*/best.pt`。
  epochs：30、41、47、46、53。
- η=1 对照：`/data2/yb/remote_experiments/osram_forward_only_mosi_20260908/seed_*/best.pt`。
  两组沿用各自已有 checkpoint，不根据本次 predictor 指标重新选 epoch。
- 两组均为已有 Test-oracle-selected 模型，因此本分析仍是内部诊断。
- 10 个 checkpoint strict loading；5 对 mask SHA256 和 conversation ID 顺序完全一致。
- 所有有效 utterance 均参与，不因情绪标签为 0 而排除：这是 latent audit，不是二分类 F1。
- regression、contrastive 两头分别评估，不能用后者替代实际拟用于补全的 regression latent。
- 直接复用旧 `missing_m3_mosi_latent_diagnostic_20260831/analyze_checkpoint.py::_metrics`。
  Shuffle 在 **同 seed / pattern / target** 内执行 8 次固定随机置换；与旧版一致，置换允许少量固定点。
- centered cosine 分别减去当前子群 prediction / teacher 的样本均值。
  Retrieval 使用该子群全部 teacher 为候选、centered cosine 最近邻，chance=1/N；不是跨模态负样本池。
- Effective rank 沿用旧版 **中心化矩阵奇异值熵**，不是协方差特征值熵；上限为 min(N−1,256)。
  不把不同群体 N 不同的 rank 当作完全同条件比较，也不混合多个 seed 的 teacher 空间。
- Std ratio = mean-channel population std(pred) / mean-channel population std(teacher)，归一化前计算。
- 所有表格先逐 seed 计算再等权平均；[完整表](TABLES.md) 与 [summary.csv](summary.csv) 含 sample SD。

## Text regression prediction：η=.6 五种子均值

| Observed→target | Raw cosine | Centered cosine | Real−Shuffle cosine | Retrieval / chance | Pred / teacher rank | Std ratio |
|---|---:|---:|---:|---:|---:|---:|
| A→T | .93095 | .03785 | .00094 | 1.020% / .847% | 7.57 / 76.37 | .126 |
| V→T | .92768 | .02862 | .00060 | 1.743% / .858% | 11.53 / 76.71 | .112 |
| AV→T | .92948 | .04262 | .00085 | 1.431% / 1.196% | 9.39 / 60.12 | .111 |

Text prediction 有少量变化，但幅度只有 teacher 的约 11%–13%，rank 也远低于对应 teacher。
V→T retrieval 均值高于机会水平，不能称为严格零信息；但这不等于可靠补全或情绪增量。

## 与 η=1 同 mask 对照

| Observed→Text | Centered η=1 | Centered η=.6 | Centered 提升 seeds | Real−Shuffle 提升 seeds |
|---|---:|---:|---:|---:|
| A→T | .05651 | .03785 | 1/5 | 1/5 |
| V→T | .05199 | .02862 | 1/5 | 1/5 |
| AV→T | .05737 | .04262 | 2/5 | 1/5 |

两套模型 teacher 本身也随训练改变，因此这是“各自训练目标空间内的辨识表现比较”，
不是把两套 latent 向量直接对齐，也不能仅由此证明 η 对 predictor 的直接因果机制。
旧 20260831 audit 使用另一模型、rate 与 checkpoint-selection 协议，仅继承指标，不能冒充严格配对结果。

## 其他目标与 contrastive 分支

- A→Visual regression centered cosine .24217，Real−Shuffle .02457，retrieval 2.156% / chance .847%。
  这一路样本对应信号明显强于 Text 方向，但 rank 仍仅 5.53 / teacher 49.16，不能泛称所有方向全无信息。
- V→Audio regression centered cosine .00945，retrieval .190% / chance .858%，仍很弱。
- Contrastive Text centered cosine .033–.041，retrieval 也低；相比 η=1 的变化方向不统一。
  即使 contrastive 改善，也不能据此认定 regression completion 被修复。

## 后续回灌应保留的控制，而非禁止回灌的门槛

本次不训练这些控制。若后续 emotion loss 联合塑造回灌 predictor，至少保留：

1. 不回灌的当前模型 reference。
2. 相同回灌结构、用仅由训练集估计的 target prototype 替代样本预测：区分原型先验与样本增量。
3. 同 target / pattern 内打乱预测与 utterance 对应关系的诊断：区分样本身份与仅有分布/幅度作用。
4. 在联合训练后重新运行本次 audit；不能用“现在预测弱”推断联合训练后一定无效。

这些控制不应混入标签或 test target 内容。当前结果也不能把 miss=.5 分类下降归因于
“错误 latent 回灌”：目前分类路径根本未回灌 predictor。

## 验证与运行

- 100 个 seed×variant×pattern×target×branch 组完整，prediction/teacher 全有限。
- 10 次评估检查启用 predictor 前后 logits 逐元素相同，state_dict 前后逐元素相同。
- 模型 eval + torch.no_grad，无 optimizer / EMA update；在 biggpu 官方 Python 上 CPU 单线程执行。
- 原始逐 seed JSON 含 checkpoint 路径、epoch、完整配置、mask SHA256、会话顺序及全部指标。
- [逐 seed 指标](results/per_seed.csv)、[配对变化](paired_delta.csv)、[完整均值±SD](TABLES.md)。

```bash
cd /data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
 /data2/yb/reproduction_envs/s0/bin/python3.10 \
 experiments/osram_predictor_reaudit_20260909/audit.py \
 --root /data2/yb/remote_experiments \
 --feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features \
 --output /data2/yb/remote_experiments/osram_predictor_reaudit_20260909
```

将输出置于本目录 `results/` 后运行 `python summarize.py` 可重建汇总，不再运行模型。
