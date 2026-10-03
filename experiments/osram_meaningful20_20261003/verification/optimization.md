# Optimization-family verification / 优化族验证

Implemented three independent mathematical adaptations; no external implementation was copied. References and explicit adaptation boundaries are recorded in `optimization.json`.

实现三项独立数学改写，未复制外部实现；论文来源及适配边界见 `optimization.json`。

| ID | Complete computation / 完整计算链 | Parameters / 参数量 |
| --- | --- | ---: |
| `hamburger_nmf_full` | Shared head tokens → lower bread/ReLU → rank-4 NMF, six detached iterations plus one differentiable coefficient update → upper bread/LN/residual/ReLU → typed difference means → 128d | 158080 |
| `crate_mssa_ista_full` | Shared head tokens → two complete PreNorm tied-subspace attention + dictionary ISTA iterations, without an extra ISTA residual → typed difference means → 128d | 224512 |
| `equilibrium_aggregation` | Shared head tokens → learned squared residual potential plus aggregate regularizer → ten differentiable Nesterov updates from zero → 128d aggregate | 183491 |

Counts use Local=256, heads=8, head dimension=64, excluding the integration-owned 128→1600 zero-initialized bridge. Hamburger's positive basis is a fixed buffer: inferred factors reset on every call and never update persistent state. EA's inner objective defines its forward computation, not an additional training loss. No OSRAM read, write, query, or cache is added.

参数量按 Local=256、8 个 64 维 head 计算，不含集成层负责的 128→1600 零初始化桥接。Hamburger 的正基矩阵为固定 buffer，每次调用重新推断因子，不修改持久状态。EA 内层目标仅定义前向计算，并非额外训练损失；不新增 OSRAM 读、写、query 或 cache。

## Evidence / 验证证据

Test-first run failed for all five initial tests before implementation. Final command:

先写的五项测试在实现前全部失败。最终验证命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_optimization.py
```

Result: **8 tests passed**, 0.221 seconds on CPU. Coverage:

结果：CPU 上 **8 项测试通过**，耗时 0.221 秒，覆盖：

- Independent numerical references for CRATE's complete update, NMF forward and one-step backward, EA's full potential/energy, and ten-step Nesterov on a quadratic energy.
- 独立数值参考：CRATE 完整更新、NMF 前向与单步梯度、EA 完整势函数/能量及二次能量上的十步 Nesterov。
- Finite input/parameter gradients, inactive NaN masking, all-inactive and empty batches, per-row batch independence.
- 输入/参数梯度有限、inactive NaN 屏蔽、全 inactive 与空 batch、跨 batch 样本独立。
- Evaluation under `inference_mode`, unchanged state dictionaries/RNG, no evaluation parameter-gradient accumulation; private deterministic basis and factory RNG isolation.
- `inference_mode` 下评估，状态字典/RNG 不变且不累积参数梯度；私有确定性基矩阵及工厂 RNG 隔离。

Real-dimension CPU smoke with three rows completed forward/backward with finite outputs and input gradients. Observed single-call times: Hamburger 0.0214 s; CRATE 0.0111 s; EA 0.0734 s. These are smoke timings, not benchmark claims.

真实维度、三个样本的 CPU 前向/反向冒烟通过，输出和输入梯度有限。单次耗时分别为 0.0214、0.0111、0.0734 秒，仅为冒烟记录，不是性能基准。

## Limits / 未验证事项

No real-data training, GPU integration, end-to-end peak memory, or sentiment score has been tested in this lane. EA retains a ten-step higher-order graph during training and needs integration-level GPU feasibility checking before queueing full runs. Source-specific task heads/losses are intentionally excluded; the original task-only objective remains owned by integration.

本工作支线未执行真实数据训练、GPU 集成、端到端峰值显存或情感分数测试。EA 训练保留十步高阶计算图，正式入队前需验证 GPU 可行性。来源方法的任务头/损失不迁入，集成层继续沿用原 task-only 目标。
