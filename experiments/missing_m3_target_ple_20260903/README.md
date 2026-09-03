# Missing-M3 Target-Private Expert Residual

## 版本名称

- Git 分支：`feature/missing-m3-target-ple`
- 方法简称：`TPER-M3`
- 代码目录：`gcnet_missing_m3/`
- 核心开关：`--target-private-rank`

## 解决的问题

原 Missing-M3 的 A/T/V 三类缺失目标共用同一组 MMoE experts。即使输出 head 已按目标模态区分，进入 head 之前的共享表示仍可能受到不同目标预测梯度的相互干扰。

TPER-M3 保留共享 experts，并为每个目标模态增加一个低秩私有残差：

```text
source latent + context + source/target identity
                    │
          ┌─────────┴─────────┐
          │                   │
     shared MMoE       target-private expert
          │                   │
          └─────────+─────────┘
                    │
        target-specific reg/cl heads
```

私有专家为 `Linear(d,r,bias=False) -> GELU -> Linear(r,d,bias=False)`。Audio、Text、Visual 各自拥有一套参数；某个目标的损失不会更新其他目标的私有专家。末层零初始化，因此训练开始时与原共享预测器输出一致。

## 配置

- Control：`--target-private-rank 0`
- Treatment：`--target-private-rank 32`
- 当 `latent_dim=256`、`rank=32` 时新增参数：`3 * 2 * 256 * 32 = 49,152`
- 不启用 `classification_completion` 时，该模块仅用于训练期 JEPA，测试阶段没有额外计算。

## 后续正式实验（本提交未启动）

MOSI，seeds 66–70，一个 all-rates checkpoint 测试 0.0–0.7 八个 missing rates；Control 继承已有结果，不重跑 Original。第一轮只比较 `rank=0` 与 `rank=32`，不叠加 Contrastive Fusion。

本轮按要求只交付完整代码、配置、测试与版本说明，不运行 smoke 或正式训练。
