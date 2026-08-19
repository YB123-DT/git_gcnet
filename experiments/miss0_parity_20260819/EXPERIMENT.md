# GCNet miss=0 严格等价性检查

## 目的

验证 Original GCNet 与 Modality-JEPA 在 `missing_rate=0` 时是否存在隐藏的
JEPA 训练路径。两者从同一份 GCNet state 初始化，使用固定顺序的同一批
IEMOCAP-Six 数据、全 1 预生成 modality mask，以及 seed 66/67/68。

每个 seed 连续运行三个 optimizer steps，并记录：

- `max |logits_base - logits_candidate|`；
- reconstruction、total loss、共享参数 gradient 的最大差；
- Adam step 后共享权重的最大差；
- JEPA loss、JEPA loss 对模型参数的梯度范数；
- Predictor 梯度范数。

使用 GPU0；GPU4 未使用。实际特征维度为 Audio 512、Text 1024、Visual
1024，batch size 为 32。

## 结果

### 初始单步 parity

| Comparison | Seeds | Initial shared state | Step-1 logits max abs | Step-1 shared grad max abs | JEPA loss | Predictor grad |
|---|---|---:|---:|---:|---:|---:|
| Baseline vs JEPA | 66/67/68 | 0 | 5.40e-8–6.71e-8 | 7.45e-9–1.49e-8 | 0 | 0 |
| Baseline vs Baseline | 66/67/68 | 0 | 5.59e-8–6.71e-8 | 7.57e-9–1.49e-8 | 0 | 0 |

初始 logits 均满足 `<1e-6`。`missing_rate=0` 时 Predictor 没有执行，
JEPA loss 是 detached scalar zero，不进入 total loss，也没有任何参数梯度。

### 连续 optimizer steps

| Comparison | Step-1 后权重差 | Step-2 logits 差 | Step-3 后权重差 |
|---|---:|---:|---:|
| Baseline vs JEPA | 2.46e-7–3.56e-7 | 0.0996–0.1340 | 3.51e-4–3.54e-4 |
| Baseline vs Baseline | 2.03e-7–4.05e-7 | 0.1119–0.1243 | 3.51e-4–3.52e-4 |

两个纯 Baseline 副本表现出与 Baseline-vs-JEPA 同量级的漂移。因此连续
step 失败不是 Predictor、JEPA loss 或 JEPA model class 引入的，而是两份
GCNet CUDA 训练副本本身不能维持逐 bit 相同的轨迹。首步约 `1e-8` 的
gradient 差经 Adam 更新及后续 recurrent/graph computation 被快速放大。

## 结论与实验规则

此前 `missing_rate=0` 的 Original 0.61729 与 JEPA 0.60296（-1.43 W-F1
points）来自两次独立训练，不能解释为 JEPA 的负作用。因为该条件下辅助
目标严格关闭，它测到的是 GCNet/PyG CUDA 训练轨迹噪声。

后续采用以下规则：

1. `missing_rate=0` 不再独立训练一份“JEPA”模型；直接复用同一个 Original
   checkpoint、logits 和指标，因此 paired difference 按定义为 0。
2. 只有 `missing_rate>0` 才执行 Predictor 和 temporal/modality loss。
3. 非零 missing rate 的结论必须来自相同 seed 的多次重复及配对差，而不是
   单次运行。
4. 代码级 parity gate 保留：同 checkpoint 的初始 logits `<1e-6`，
   `L_JEPA=0`，JEPA gradient norm=0，Predictor gradient norm=0。

原始机器可读记录：`parity.json` 与 `baseline_control.json`。

## 命令

```bash
CUDA_VISIBLE_DEVICES=0 /data2/yb/reproduction_envs/s0/bin/python \
  scripts/run_miss0_parity.py \
  --comparison baseline-jepa --seeds 66 67 68 --steps 3 \
  --output experiments/miss0_parity_20260819/parity.json

CUDA_VISIBLE_DEVICES=0 /data2/yb/reproduction_envs/s0/bin/python \
  scripts/run_miss0_parity.py \
  --comparison baseline-baseline --seeds 66 67 68 --steps 3 \
  --output experiments/miss0_parity_20260819/baseline_control.json
```
