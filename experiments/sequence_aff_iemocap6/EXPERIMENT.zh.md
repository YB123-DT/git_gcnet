# Mask-conditioned Sequence AFF 探究记录

## 研究问题

GCNet 的两个关系图分支分别产生 `hidden1` 与 `hidden2`，官方实现直接执行：

```python
hidden = hidden1 + hidden2
```

本实验考察：Dai et al.（WACV 2021）的 Attentional Feature Fusion（AFF）能否在不改变图结构、RGCN、第二层 GraphConv、BiLSTM、重建损失和分类器的条件下，根据当前话语内容及缺失模式自适应融合两个图分支。

## 方法与理由

对两个图分支先求和 `S = hidden1 + hidden2`。使用话语级局部上下文和 conversation 内、由 `umask` 约束的全局时间均值共同生成通道门控：

```text
w = sigmoid(local([S, pattern]) + global(masked_mean([S, pattern])))
A = 2 * (w * hidden1 + (1 - w) * hidden2)
H = where(complete, S, A)
```

其中 `pattern` 是 A、T、V、AT、AV、TV 六维编码，ATV 编码为全零。使用 `2×` 缩放是为了使 `w=0.5` 时严格恢复直接加法。局部和全局门控的末层零初始化，因此训练开始时所有缺失模式也与加法一致；完整模态通过 `where` 永久严格恢复官方加法。

选择这一替换的逻辑是：两个图分支提供不同关系语境，直接相加默认其贡献恒等；AFF 可以在通道级根据内容、序列上下文和缺失模式调节两者贡献，同时保持完整模态行为不变。

## 改动边界

仅替换 `GraphModel.forward()` 中的 `hidden1 + hidden2`。以下部分保持不变：

- 输入特征、缺失 mask bank 与 missing-rate 生成；
- speaker/temporal 图拓扑和 relation；
- 两层图卷积与图后 BiLSTM；
- MatchingAttention、重建分支、分类器与损失；
- optimizer、训练轮数、fold 和随机种子。

## 验证分层

验证已拆成互不混跑的四层：

1. 数学单测：7 种 pattern、完整模态前向/梯度恒等、零初始化、内容条件和 padding；不训练。
2. 接线测试：默认 addition 的参数/RNG/输出不变，AFF 只接管分支融合；不训练。
3. 保存集成：只有训练或 NPZ 保存代码变化时才执行一次 1 epoch。
4. 正式实验：代码冻结后单独运行；禁止夹带 smoke，Original 优先继承。

常规无训练验证由原先分散、重复的 27 项缩减为 14 项，运行时间 1.558 秒。清理计划见 `docs/superpowers/plans/2026-08-25-sequence-aff-validation-cleanup.md`。

## 锁定实验协议

- 数据集：IEMOCAP-6；
- fold：5；
- missing rate：0.0–0.7，共 8 档；
- seeds：66–70，共 5 个；
- 每个方法 40 个任务；
- Sequence-AFF：40/40 正式任务成功；
- GPU：biggpu 0–3，每卡 3 并发；
- 训练/验证/测试均读取既有固定 stage-aware mask bank；
- 每个 Original/Sequence-AFF pair 的 mask SHA256 完全一致。

Original 没有再次训练。继承来源为：

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
protocol_recovery_v1_biggpu/formal/original
```

旧档案产生时尚未增加 `pre_graph_context`、`post_graph_context` 和 `branch_fusion` CLI 字段；旧代码对应的唯一行为是 `bilstm/bilstm/addition`。其参数量 `34,140,166` 与当前 addition 的实际选中路径参数量一致，且默认路径的参数、RNG 和输出恒等测试已通过。

Sequence-AFF 结果来源为：

```text
/data2/yb/paper/experiments/sequence_aff_iemocap6_20260825_v2/
formal/sequence_aff
```

## 结果

| 缺失率 | Original F1 | Sequence AFF F1 | 配对差值 | 胜/平/负 | 配对 t 检验 p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.630557 | 0.625325 | -0.005232 | 2/0/3 | 0.263417 |
| 0.1 | 0.636806 | 0.636571 | -0.000235 | 2/0/3 | 0.977997 |
| 0.2 | 0.642733 | 0.644052 | +0.001319 | 3/0/2 | 0.657737 |
| 0.3 | 0.636027 | 0.637686 | +0.001659 | 3/0/2 | 0.907054 |
| 0.4 | 0.635084 | 0.619337 | -0.015747 | 1/0/4 | 0.188567 |
| 0.5 | 0.613622 | 0.618554 | +0.004932 | 2/0/3 | 0.584849 |
| 0.6 | 0.617032 | 0.593800 | -0.023232 | 1/0/4 | 0.142401 |
| 0.7 | 0.603929 | 0.609480 | +0.005551 | 3/0/2 | 0.495040 |

八档宏平均：Original `0.626974`，Sequence-AFF `0.623101`。按 seed 先跨 missing rate 平均后的配对差值为 `-0.003873 ± 0.006432`，胜/平/负为 `2/0/3`。

机器可读结果见 `results/summary.json`，自动生成表见 `results/RESULTS.zh.md`。

## 结论

Sequence-AFF 未通过主方案筛选。它在 0.2、0.3、0.5 和 0.7 有小幅正增益，但所有单档差异均不显著，并在 0.4、0.6 明显下降，最终宏平均低于 Original。

结果说明 speaker/temporal 两个图分支并不适合仅凭当前融合状态和 missing pattern 进行通道竞争：门控在不同缺失率上的方向不稳定，而且在 complete setting 的严格保护无法阻止缺失训练对共享上游表示产生间接影响。该模块保留为负结果/消融候选，不作为 GCNet 的主优化模块。

## 后续执行规则

- 新模块共用本实验的锁定 Original，不再重复训练 baseline；
- 只在 Original 数据、mask、随机轨迹或实际计算路径发生变化时重训；
- 不为每个模块复制 runner、汇总器和测试框架；
- 下一项 BiLSTM 消融继承 `BiLSTM/BiLSTM` baseline，只运行另外三个组合。

