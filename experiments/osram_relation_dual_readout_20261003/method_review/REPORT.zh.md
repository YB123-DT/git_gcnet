# Relation 之后该怎么改：论文、代码与实验约束

INTERNAL DIAGNOSTIC ONLY

日期：2026-10-03；本地证据版本 d2dc748。本次只整理，不实施新模块，
不做模型推理、模型梯度审计、训练或调参。论文方法节／相关限制与指定
作者代码均按下表核查；没有复现外部论文完整实验。文献溯源技能要求
保留来源和中英文记录，设计技能使本轮停留于方案讨论。

## 结论

目前没有找到能够承诺 OSRAM 提分的方法。值得列为**有条件候选**的是：
保留 Relation + 双读出的前向结构，借用 PCGrad 协调两项任务梯度。
这是一项训练更新规则，不是新 Memory、Gate 或缺失模态补全 Block。
不能把它包装成独立的新架构贡献，也不能称其解决了全程缺模态的信息不足。

与上一轮无条件 detach 提议相比，它只在两个梯度反向时介入；因此需要
先检查实际梯度，而不是根据性能下降推断存在梯度冲突。两种改动不叠加。
这不是已证实优于 detach 的结论，后者也未实施或训练。

## 已有证据能支持什么

- [三种子双读出结果](../MULTISEED.md)：八率 80.315，原 Flat 80.559；
  高缺失 75.366 对 75.594。当前候选不是强于原 Flat 的新基线。
- [单损失 Relation 的 seed66 residual-off 诊断](../../osram_current_history_relation_20261003/residual_off_analysis/RESULT.md)：
  R-off - Flat 为 -0.707，R-on - R-off 为 +0.216 pp。
  分支即时贡献与训练后原路径变化不同，值得研究联合训练；**不能据此
  证明双读出三种子存在梯度冲突或证明冲突导致下降**。
- [原 Gap audit](../../osram_gap_increment_audit_20261003/RESULT.md)表明
  历史总体有用，但简单内部量难以稳定区分 rescue/harm。事后找到错误
  不等于新 Gate 在推理时能识别错误。

上述都是 Test-oracle 内部诊断。选候选也参考了这些测试结果，不能把
后续测试提升当成无偏选模结论；正式结论仍需固定 validation 规则。

## 文献筛选：不是只看名称或摘要

| 方法 | 原文的核心做法 | 当前适配判断 | 核查范围 |
|---|---|---|---|
| [Side-Tuning, ECCV 2020](https://arxiv.org/pdf/1912.13503) | 固定预训练 base，学习旁路表示 | 原方法与从头一阶段、主干正常训练的要求不符，不直接采用 | §3.1、§4.4–4.5；作者 `tlkit/models/sidetune_architecture.py` 的 base/side merge |
| [Gradient Similarity](https://arxiv.org/pdf/1812.02224) | 用主／辅助任务梯度相似性调节辅助更新 | 可解释风险，但不能把将要部署的 Full 轻率地当成可丢弃辅助任务 | §2–3、Proposition 1 后的限制；未核实作者实现，不作为代码来源 |
| [PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | 冲突时投影任务梯度，未冲突时保留 | 本轮优先的有条件候选：两读出均保留，无额外混合超参 | §2.2–2.4、Algorithm 1、§3；作者 `PCGrad_tf.py` |
| [CAGrad, NeurIPS 2021](https://arxiv.org/pdf/2110.14048) | 在平均梯度附近优化最弱任务的局部改善 | 有理论动机，但引入 c 和内部优化求解，本轮不同时增加 | §3.1–3.2、Algorithm 1；作者 `toy.py::cagrad`，并查看 NYUv2 utils |

作者代码链接：
[PCGrad](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)、
[CAGrad](https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py)、
[Side-Tuning](https://github.com/jozhang97/side-tuning/blob/master/tlkit/models/sidetune_architecture.py)。
Side-Tuning 的冻结约束以论文 §3.1 为准；读取的通用 merge 类不是冻结
实现的完整追踪。作者代码链接是分支链接，不声称已固定外部 commit。

## 映射到 OSRAM：保持前向不变

这里的 `base readout` 指完整 Local + Memory Flat，不是只保留 OSRAM Base。

```text
一份输入、一次 causal OSRAM scan → Local、Base、masked Gap
u = LocalSkip(Local) + Adapter([Local, Base, masked Gap])
r = existing_Relation(Local, Base, masked Gap, availability, umask)
pred_base = Head(LN(u))
pred_full = Head(LN(u + r))
```

仍从头一阶段联合训练；不冻结、不 detach、不增加 loss，仍计算原
0.5 task(base) + 0.5 task(full)。改动仅是如何合并梯度，因此不能把
新优化过程称为与原标量 loss 的普通反向传播完全等价。

参数分为 θ（两个读出共享的原可训练参数）和 φ（Relation 专属参数）。
对原来未乘 0.5 的两项任务损失，定义：

```text
g_b = grad_theta(task_base)
g_f = grad_theta(task_full)
d   = dot(g_b, g_f)
```

当 d >= 0 时，沿用 `(g_b + g_f)/2`。
当 d < 0 且范数非零时，使用两份**原始梯度**同时计算：

```text
p_b = g_b - d / ||g_f||² * g_f
p_f = g_f - d / ||g_b||² * g_b
shared_grad = (p_b + p_f) / 2
relation_grad = 0.5 * grad_phi(task_full)   # 两种情况均如此
```

全局投影只覆盖 θ，φ 不参与点积或投影。不能凭空给 Relation 分支
制造 base loss 梯度。逐层投影、只保护 base 的非对称投影属于不同变体，
不同时引入。无共享任务梯度的参数保持 None，不因填零触发额外权重衰减。
非有限值必须报错；近零范数需要明确数值处理，不能静默产生 NaN。

## 代码层面必须注意

- 作者 PCGrad 实现是 TensorFlow，不是可直接粘贴的 PyTorch 优化器。
  它将非 None 梯度展平；本项目 base/full 的参数依赖不同，必须固定 θ
  的名称、顺序和形状，显式区分 φ，不能分别丢掉 None 后直接做点积。
- 作者实现合并为求和；本项目固定两项各 0.5，必须明确取平均，避免
  把梯度放大两倍误认为方法变化。零 residual 时共享梯度应恢复原双读出。
- `_relation_dual_readout_loss` 提供两读出任务监督；普通 `loss.backward()`
  后紧跟梯度裁剪和 optimizer.step。若未来实施，应在裁剪前完成合并，
  保留 Adam、lr、weight decay 和原 clipping 配置，而不是同时换优化器。
- 一次前向／一条 Memory 轨迹不等于训练开销不变：分别求两项梯度需要
  额外反向计算和图保存，应记录耗时、显存；不能宣称免费或恰好两倍。
- 推理仍只有 Full 输出，无第二视图、无梯度计算，不依赖测试 batch 中
  别的样本拥有缺失模态，不引入历史缓存、额外 query 或补全写入。

## 不能承诺的事情

PCGrad 原文明确讨论：负梯度内积本身不足以证明有害，幅值与曲率也相关。
Gradient Similarity 原文同样指出局部相似性不保证正迁移。
本项目还是同一情感任务的两个嵌套读出，不是论文中的独立多任务基准；
梯度可能高度同向，也可能受 dropout、mask 和 minibatch 噪声影响。

两任务精确投影的合成方向，在普通欧氏空间有局部一阶相容性；它不是
Adam 动量／预条件和 weight decay 后的下降保证，也不保证 MSE 或 W-F1
在有限步、验证集、测试集上改善。完全反向梯度可使共享任务更新归零。
参数仍会变化，不能称原 Flat 功能或 Memory 轨迹被严格保住。

## 唯一 NEXT ACTION：先做不更新权重的训练梯度审计

若继续实施，先复用已有 Relation dual checkpoint，只用允许的训练
conversation 和训练标签，在固定 batch／missing masks 上计算 g_b/g_f：

- cosine 与冲突次数／总 batch 数；
- 两项梯度范数比；
- 投影前后合成梯度范数比和方向变化，极端反向／零范数计数；
- 按 seed、checkpoint 对应 rate 和 batch 分开记录，重复固定抽样
  检查稳定性，不用一个平均值掩盖异质性。

训练模式下共享同一次 dropout forward；使用隔离加载的模型，固定并
保存 RNG/buffer 状态，禁止 optimizer.step，核对参数和 checkpoint
前后 hash。只看现有 checkpoint 不能代表整个训练历史，必须保留该限制。
测试标签、near/same/opposite 标签都不进入审计决策。

若大多同向、冲突仅偶发，或投影大量消除更新，则不据此启动 PCGrad。
即使冲突稳定，也仅说明值得进一步受控验证，不证明它是根因。
不在本次同时运行 detach、PCGrad、CAGrad 多组试验。

## 本次已验证与未验证

已读原文指定章节、作者指定代码以及本地双读出损失／更新位置。
离线纯向量代数检查 1000 对随机向量（seed66）：510 对冲突、490 对
同向；检查合并方向、非冲突不变、反向归零，并构造对角预条件可改变
方向相容性的反例。这不是模型测试，不涉及数据集或任何模型 forward。
未测 OSRAM 梯度分布，未实现 PCGrad，未启动实验，未证明性能提升。
