# MOSI Budget-Preserving Sparsity-Weighted JEPA 设计规格

## 1. 诊断依据

Shared Post-Graph BiLSTM 的五种子确认失败：overall `-0.1785` point、
high-missing `-0.6508` point，且 strong control seed 69 下降 `2.3277` points。
因此关闭 graph-post hard/partial sharing，不在同族结构中继续搜索。

当前已有独立五种子消融证明，原 Missing-M3 JEPA 相对 `lambda_J=0`：

- overall `+0.4559` point，`4/5` seeds 为正；
- high-missing `+0.6593` point，`4/5` seeds 为正；
- 不增加测试时模块。

说明 JEPA 是目前少数已有稳定高缺失正向证据的机制。下一候选不增加模型容量，
只重新分配八-rate 同批训练中的辅助梯度预算。

## 2. 唯一变量

新增：

```text
jepa_rate_weighting = uniform | sparsity-budget
```

默认 `uniform` 严格保持：

```text
loss_eta = task_eta + lambda_J * jepa_eta
```

候选 `sparsity-budget` 使用：

```text
loss_eta = task_eta + lambda_J * w_eta * jepa_eta
w_eta = (1 + eta) / 1.4
```

在 `eta=0` 时没有 missing target，`jepa_eta` 精确为零。七个实际有 JEPA target 的
rates `0.1--0.7` 满足：

```text
mean(w_0.1, ..., w_0.7) = 1
```

因此 active-rate 的 JEPA 系数预算仍精确等于 `lambda_J=0.1`；它只把低缺失条件的
一部分辅助梯度转移给高缺失条件，不提高总名义预算。

## 3. 保持不变

- frozen wav2vec/DeBERTa/MANet features；
- Student、EMA Teacher、MMoE、GCNet 和 classifier；
- task MSE、`lambda_J=0.1`、InfoNCE/SmoothL1 内部公式；
- all-rates-per-batch 的八次 forward 与 `1/8` backward；
- natural masks、splits、relation mapping、optimizer 和 checkpoint selection；
- 推理路径、参数量、计算量与显存。

不得与 Shared BiLSTM、SmoothL1 task、Packed recurrent、conditioned readout 或
utterance-balanced JEPA 组合。

## 4. 实验门槛

先运行 MOSI hidden100/window1、seeds 66--68、validation-only，并继承 direct
deterministic Legacy controls。沿用锁定 gate：overall delta 至少 `+0.40` point、
至少 `2/3` seeds 为正、high-missing 非负、miss-0 不低于 `-0.30` point、最差 seed
不低于 `-1.0` point且无坍塌。

只有通过才补 seeds 69/70；五种子要求 overall 正、至少 `4/5` seeds 正、
high-missing 非负、miss-0 不低于 `-0.30` point且无坍塌。此前不读取 test。

## 5. 失败边界

若失败，说明简单 rate-level JEPA 预算转移不足，或高缺失 latent target 的歧义噪声
抵消了增益。不扫描权重曲线、不与其他候选组合。下一独立候选才考虑保留双图独立
BiLSTM 的 branch-specific graph-message calibration。
