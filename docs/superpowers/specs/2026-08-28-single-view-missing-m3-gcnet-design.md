# Single-View Missing-M3 GCNet 设计规格

## 1. 研究问题

本实验只回答一个问题：能否从同一个官方不完整输入中构造统一的 utterance node，交给 GCNet 建模对话关系，并用训练期 M3 式方向预测增强其对真实缺失模态的表征？

本版本不使用 Natural/Pattern 双视图，不执行测试时模态恢复，也不保留 Original linear reconstruction。

## 2. 主路径

```text
official incomplete A/T/V
→ observed modality projectors
→ observed-set fusion
→ Original GCNet temporal/speaker context backbone
→ emotion classifier
```

对 utterance `i` 的可见集合 `O_i`，模态投影为：

\[
s_i^m=P_m(x_i^m),\qquad m\in O_i.
\]

缺失模态不会执行有效 projector 计算，其输出在融合前由 availability 显式置零。加入模态身份和可见集合身份后：

\[
u_i=\rho\left[
\frac{1}{|O_i|}\sum_{m\in O_i}(s_i^m+e_m)+e_{O_i}
\right].
\]

`u_i` 替换 Original GCNet raw feature concatenation，后续保留 GCNet 的 pre-graph recurrent encoder、Temporal graph、Speaker graph、graph-post sequence encoder、branch addition 与 classifier。

因此最终主干是 GCNet；Observed-Set Fusion 只是 node construction，不是另一个 backbone。

## 3. 训练期 Missing-Latent Predictor

Missing-Latent Predictor 是 M3 风格的六方向 Top-K MMoE。它接收当前可见 source latent 和同一次 GCNet forward 的上下文：

\[
r_i^m=\operatorname{LN}(s_i^m+W_hh_i),
\]

\[
(\widehat z_{i,q}^{reg},\widehat z_{i,q}^{cl})
=F_{m\rightarrow q}^{M3}(r_i^m),\qquad m\in O_i,\ q\notin O_i.
\]

M3 内部保留：

- source/target identity embedding；
- shared experts；
- Top-K routing；
- regression gate 与 contrastive gate；
- target-specific output heads。

当两个 source 都可见时，同一 target 的方向预测取平均：

\[
\widehat z_{i,q}=\frac{1}{|O_i|}\sum_{m\in O_i}F_{m\rightarrow q}^{M3}(r_i^m).
\]

实现按六个方向整批向量化，不允许逐 utterance Python predictor 循环。

## 4. EMA Target Encoder 与损失

完整训练特征只在无梯度 teacher 分支中产生真实缺失目标：

\[
z_{i,q}^{t}=\operatorname{stopgrad}(P_q^t(x_i^q)).
\]

Teacher 在 optimizer step 后更新：

\[
\theta_q^t\leftarrow\tau\theta_q^t+(1-\tau)\theta_q^s.
\]

总损失仅包含：

\[
\mathcal L=\mathcal L_{emotion}
+\lambda_J\left(\tfrac12\mathcal L_{SmoothL1}
+\tfrac12\mathcal L_{InfoNCE}\right).
\]

当一个 target group 少于两个有效样本时，只计算 SmoothL1，跳过该组 InfoNCE。Padding、observed target 和完整 ATV utterance 不进入 JEPA loss。

## 5. 测试路径

测试时删除 M3 Predictor 与 EMA Teacher：

```text
actual incomplete input
→ observed-set fusion
→ GCNet
→ emotion prediction
```

不生成缺失 latent，不回灌预测，不增加推理分支。

## 6. Mixed-rate 协议

首轮锁定：

- 数据集：IEMOCAPSix；
- held-out fold：Session 5；
- seeds：66、67、68、69、70；
- rates：0.0、0.1、0.2、0.3、0.4、0.5、0.6、0.7；
- 每个 seed 训练一个 checkpoint；
- 每个 epoch 的 train batches 均衡轮换八个 rate，epoch 间循环起点平移；
- 每个 rate 使用 conversation-keyed deterministic mask schedule；
- checkpoint 按八个 validation rate 的等权平均 weighted F1 选择；
- 选中 checkpoint 后分别测试八个固定 rate；
- Original 不为本轮重新训练。

总训练任务数为 5，而不是 `8 rates × 5 seeds = 40`。

## 7. 最小验证

实现后仅执行与风险匹配的验证：

1. 七种非空 availability 均产生有限 node；缺失 block 数值变化不影响融合结果。
2. 六方向 M3 routing、双 source 平均和 target selection 正确。
3. Teacher 无梯度，EMA 必须发生在 optimizer step 后。
4. Predictor 关闭时仍能完成八个 rate 的分类 forward。
5. 一个 GPU batch 的 forward/backward 通过后立即启动正式 5-seed 队列，不运行重复 smoke。

## 8. 解释边界

可以声称：一个 observed-set node construction 与 GCNet conversation backbone 通过训练期 M3 方向预测联合学习统一的不完整表示。

不能声称：严格恢复 Original GCNet、测试时恢复缺失模态、显式执行两个 GCNet view，或 M3 Predictor 属于推理主干。
