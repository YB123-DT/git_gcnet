# Independent hypergraph review / 超图独立审查

Final decision after owner correction: **APPROVE all four for CPU integration within the declared adaptation boundary. No training approval.** The original HIGH finding and its resolution are retained below.

修正后结论：**四项均可进入 CPU 集成，不授权训练。** 下方保留原 HIGH 问题和修复证据。

## Resolution / 修复复核

The owner corrected the leading term to per-node/per-stalk `D^-1/2`, synchronized the card, and added both the explicit `[1.5,0.5]` counterexample and an independent sequential-source forward/map-gradient/input-gradient reference. I inspected the changes and reran the complete hypergraph suite: **12 tests passed in 1.817 seconds**. The HIGH blocker is resolved. This is CPU/core approval only; GPU feasibility and complete readout integration remain unverified here.

作者已修正首项、同步卡片，补充反例和按作者顺序独立计算的前向/输入及 restriction 梯度测试。独立重跑 **12 项通过，1.817 秒**，HIGH 阻塞已解决；尚不覆盖 GPU 和完整读出集成。

## Resolved HIGH: Sheaf normalization changes the identity contribution

`meaningful_blocks_hypergraph.py:sheaf_propagation` currently implements `I + A - 2 blockdiag(A)`. The card and numerical test repeat this formula, but it is not the effective operator of the pinned author implementation.

当前实现、卡片与数值测试一致使用上述公式，但没有复现作者实现的实际运算顺序。

The [pinned author layers.py](https://github.com/IuliaDuta/sheaf_HNN/blob/45a5ebc16ec4b8e865431f9035eefc7719e7e7dc/layers.py) first transforms the feature into `D^-1/2 Z` in the symmetric-normalization path, then left-multiplies by `I + Q - 2 blockdiag(Q)`, where `Q=D^-1/2 H B^-1 H^T`. Therefore its effective operator is:

作者先归一化特征，再应用含单位项的传播算子；因此等价算子为：

```text
A = D^-1/2 H B^-1 H^T D^-1/2
P = D^-1/2 + A - 2 blockdiag(A)
```

Independent CPU counterexample: two nodes, one edge, one stalk, both maps equal 2, transformed features `[1,3]`. Source-order calculation gives `[1.5,0.5]`; current implementation gives `[2,2]`. This is a real functional difference, not sparse/dense rounding.

独立 CPU 反例：两个节点、一个边、一个 stalk、restriction 均为 2，输入 `[1,3]`。作者顺序输出 `[1.5,0.5]`，当前输出 `[2,2]`，属于真实函数差异。

Required correction: preserve the public API but replace the identity contribution with per-node/per-stalk inverse-square-root degree; update the card/operator documentation and add a source-order regression using non-unit degree. Keep the declared epsilon stabilization. No implementation edits were made during this review.

需修正单位贡献、同步卡片并加入非单位 degree 的作者顺序回归测试；保留已声明 epsilon 稳定项。本审查未改实现。

## Other findings / 其他核对

- Hyper-SAGNN's pinned `EncoderLayer` applies `pff_n2` to the original static input, not the unused `fc2` return. Omitting that dead projection is correct. Tanh positionwise branches, dynamic-only residual, branch/final LayerNorms, leave-self-out attention and squared discrepancy are retained. [Author source](https://github.com/ma-compbio/Hyper-SAGNN/blob/69f2fbe21c455aca084497fb2d26a8207a95decd/Code/Modules.py).
- Hyper-SAGNN 删除未使用 static 投影正确；有效双分支与差异计算保留。
- AllSet retains both PMA directions, seed residual, grouped LeakyReLU-softmax, two normalizations and FFN; ED-HNN retains recipient-dependent return messages, sums, restart and shared iteration weights. Their source-specific task heads are explicitly replaced, not claimed reproduced. [AllSet](https://github.com/jianhao2016/AllSet/blob/6281a2f1a91f6f26040777bb0b2578fc035dc57a/src/layers.py), [ED-HNN](https://github.com/Graph-COM/ED-HNN/blob/fea3b8f5f11c2eb7265dd4d880e7b5796eddf958/models/edgnn.py).
- AllSet 与 ED-HNN 主要运算链符合固定卡片；未声称复现原任务头。
- Compacted per-row graphs have no cross-row normalization. All seven nonempty availability patterns are exercised. Base-only history is valid; inactive NaNs cannot enter projected active nodes. Empty/no-active rows return zero. Every hyperedge has at least two nodes, so leave-self-out attention is defined.
- 压紧后的图逐样本独立；七种有效 availability、Base-only、inactive NaN、空行及 self-exclusion 都有覆盖。

## Fresh verification / 本次验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_hypergraph.py
```

**10 tests passed in 2.036 seconds.** Existing coverage includes finite/nonzero parameter gradients, inactive input gradient zero, updates, permutation/batch independence, state/RNG stability and four core gradchecks. Passing tests do not override the source-parity blocker above. No GPU, real-data training, throughput or W-F1 was assessed.

**10 项测试通过，耗时 2.036 秒。** 已覆盖梯度、参数更新、排列/样本独立性、状态/RNG 与 gradcheck，但不能抵消上述来源一致性问题。未检查 GPU、真实训练、吞吐或 W-F1。
