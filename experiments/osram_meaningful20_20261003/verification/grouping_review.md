# Independent grouping review / 分组模块独立审查

**APPROVE for CPU integration, not training or performance claims.** No material correctness blocker found in the four declared adaptations. Reviewed implementation, fixed cards, tests, and pinned author routing sources.

**批准进入 CPU 集成，不代表训练或性能结论。** 四项已声明适配未发现实质正确性阻塞；核对了实现、固定卡片、测试和作者固定版本源码。

## Core checks / 核心核对

- Dynamic routing retains parent-specific vector votes, zero initial logits, parent-axis competition, zero-safe squash and accumulated agreement over three attached rounds. Released-code optional bias is explicitly excluded in the card. [Author source](https://github.com/Sarasra/models/blob/984fbc754943c849c55a57923f4223099a1ff88c/research/capsules/models/layers/layers.py).
- Dynamic routing 保留三轮完整投票、竞争、压缩与累积一致性，移除可选 bias 已明确声明。
- VB matrix orientation is `M_i W_ij`: the implementation contracts pose column with transform row; the author's broadcast product sums that same axis. Full-covariance scatter, mean shrinkage, positive-definite inverse scale, expected log determinant and precision quadratic agree with the stated conjugate posterior. Cholesky solves replace explicit inverse without changing the intended equation. [Author matrix votes](https://github.com/fabio-deep/Variational-Capsule-Routing/blob/78fb69dc10c71210fad8f456f1f2ce97766c4bb3/src/layers.py).
- VB 矩阵方向、完整协方差、收缩均值、期望 logdet 及二次型正确；Cholesky 解法不是额外模型改动。
- VB activation matches the explicitly selected released-code entropy form before BatchNorm: sigmoid of beta_a minus exp(Elogpi) times entropy minus beta_u. This is not asserted equivalent to the paper's mass-weighted expression. Omitting source BatchNorm is disclosed and removes cross-utterance coupling. [Author routing](https://github.com/fabio-deep/Variational-Capsule-Routing/blob/78fb69dc10c71210fad8f456f1f2ce97766c4bb3/src/vb_routing.py).
- VB 激活遵循已声明的作者代码熵版本，不冒充论文另一公式；取消 BatchNorm 的样本隔离适配已声明。
- Slot Attention uses competition over slots followed by normalization over active inputs, GRU refinement and residual MLP for three rounds. Training noise uses a private CPU generator with checkpointed state; evaluation uses a fixed saved noise buffer. Global RNG is unchanged during forwards. Training sample draw assignment depends on batching by design and is not claimed identical under regrouping.
- Slot 保留竞争、输入归一化、GRU 与残差 MLP；训练私有 RNG 可恢复，评估固定噪声。训练改变样本分组会改变噪声分配，这是明确边界。
- OT's batched masked log-domain solver retains both uniform marginals, thirty updates and paper sqrt(4) support-bin scaling. The current batched implementation matches packed per-row calculations, including empty and inactive-NaN rows. The paper-versus-author-module fixed scaling difference is disclosed. [Author Sinkhorn](https://github.com/claying/OTK/blob/2a4d3be10d305e76d55251d8f01c7f2ea7d9bf8c/otk/sinkhorn.py).
- OT 批量化仍保留双边际和多支持点聚合，逐行参考一致；论文与发布模块的固定倍率差异已声明。

## Fresh evidence / 本次证据

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_grouping.py
```

**21 tests passed in 1.753 seconds**, including the newly batched OT test. Tests cover seven patterns, inactive NaN forward/backward isolation, zero/all-inactive/empty inputs, real-head token shape, strict reload, finite gradients and updates, deterministic evaluation/batch independence, private RNG progression and replay, independent routing/statistics equations, and autocast exclusion.

**21 项测试通过，耗时 1.753 秒**，包括新批量 OT 测试；覆盖七种 pattern、NaN 隔离、空输入、真实 head、重载、梯度更新、评估独立性、私有 RNG 与数值参考。

## Limits / 限制

Only CPU tests were run here. CUDA private-state transfers, peak memory, complete OSRAM integration, full resume and task scores still need integration checks. The implementation intentionally casts statistical routing to float32; arbitrary `.double()` module execution is not a supported contract. No extra loss, reconstructed missing features, OSRAM state mutation or cross-row normalization was found.

本次仅运行 CPU 测试；CUDA、峰值显存、完整模型、完整续训与任务分数仍需集成验证。统计计算固定 float32，任意 `.double()` 模块运行不在约定接口内。未发现额外损失、补全特征、OSRAM 状态修改或跨样本归一化。
