# PLCI-JEPA 模型流程图生成提示词

## 需要提供给绘图模型的代码

只上传以下三个模型文件，不上传训练器、数据加载器、实验 runner 或结果代码：

1. `gcnet_plci_jepa/model.py`
2. `gcnet_plci_jepa/modules.py`
3. `gcnet_plci_jepa/patterns.py`

其中：

- `model.py` 定义 Natural、Auxiliary、Teacher 与 inference 共享模型路径；
- `modules.py` 定义 student projector、adapter、source anchor、graph-context
  correction、conditional innovation 和 EMA teacher；
- `patterns.py` 定义六种 auxiliary observed patterns。

## 英文正式提示词

```text
You are an expert scientific illustrator for a top-tier machine learning paper.

Read the three attached PyTorch model files carefully and draw a clean,
publication-quality vector architecture diagram for the implemented model:

"PLCI-JEPA: Source-Anchored Pattern-Lattice JEPA for Missing-Modality GCNet".

IMPORTANT GROUNDING RULES
1. Derive every module and arrow from the attached model code. Do not invent
   modules that are absent from the code.
2. Show model computation only. Do not show dataloaders, command-line options,
   mask-bank files, seeds, GPUs, epochs, logging, checkpoints, result tables,
   hyperparameter sweeps, or software engineering details.
3. Do not add contrastive learning, negative samples, attention fusion,
   test-time modality completion, generated modalities, uncertainty heads,
   MoE, GAN, KL loss, or extra graph layers.
4. The EMA teacher and target predictor are training-only. They must not appear
   in the inference computation path.
5. Raw missing modality feature blocks remain zero. The explicit pattern code
   is added after the pre-graph recurrent encoder, not into raw modality blocks.
6. Under the complete ATV condition, student adapters and pattern residual are
   bypassed in the Natural path, recovering the Original GCNet computation for
   fixed shared parameters.

CANVAS AND STYLE
- Landscape 16:9 canvas, white background, flat vector graphics.
- Use a restrained academic palette: blue for Natural/classification,
  orange for Auxiliary JEPA, purple for EMA teacher/target, green for inference,
  and dark gray for shared modules.
- Use rounded rectangles, thin consistent strokes, straight orthogonal arrows,
  generous whitespace, and a readable sans-serif font.
- No 3D effects, no glossy gradients, no decorative icons, no neural-network
  clip art, no excessive shadows, and no crowded text.
- All labels must be horizontal, correctly spelled, and large enough for a
  two-column paper figure.

LAYOUT
Create three clearly separated horizontal bands.

BAND 1 — NATURAL CLASSIFICATION VIEW (blue)
Frozen A/T/V features
→ official rate-matched natural availability mask
→ observed-modality Student Projectors P_A^s, P_T^s, P_V^s
→ zero-initialized modality residual adapters A_A, A_T, A_V
→ Original pre-graph BiLSTM
→ explicit missing-pattern embedding added to pre-graph hidden
→ Shared Original GCNet Backbone.

Inside the Shared Original GCNet Backbone, explicitly show two parallel graph
branches:
1. Temporal graph: RGCNConv → GraphConv → graph-post BiLSTM/attention;
2. Speaker graph: RGCNConv → GraphConv → graph-post BiLSTM/attention.
Merge the two branches using Addition.

From the merged GCNet hidden, draw two Natural heads:
- Emotion Classifier → L_cls;
- Original Linear Reconstruction → L_rec on naturally missing modalities.

Add a small note beside the Natural input path:
"ATV: adapters and pattern residual bypassed".

BAND 2 — PATTERN-BALANCED AUXILIARY JEPA VIEW (orange and purple)
Start from a separate copy of complete frozen A/T/V features.
Uniformly sample one observed subset for every valid utterance from:
{A, T, V, AT, AV, TV}.
Delete unobserved modality blocks
→ the same shared Student Projectors and residual adapters
→ the same pre-graph BiLSTM plus pattern hidden residual
→ exactly one auxiliary forward through the same Shared Original GCNet Backbone
→ current-view graph hidden h_J.

Draw the source-anchored target predictor as three components:
1. Source-only base anchor beta_(m→q), computed from observed source student
   latent g_m and target identity;
2. Low-rank graph-context correction c_(S→q)(h_J), conditioned on current
   observed pattern S, target q, and h_J;
3. For dual-source patterns only, bounded low-rank conditional innovation
   Delta_(n|m→q), conditioned on ordered source relation and h_J.

Show the prediction equations compactly:
Single source:
z_hat_q^{ {m} } = Normalize(beta_(m→q) + c_({m}→q)(h_J))

Dual source, two ordered paths sharing the same h_J and context:
z_hat_q^{m→n} = Normalize(beta_(m→q) + c_({m,n}→q)(h_J)
                              + Delta_(n|m→q))
z_hat_q^{n→m} = Normalize(beta_(n→q) + c_({m,n}→q)(h_J)
                              + Delta_(m|n→q))

Place the EMA teacher branch below the predictor in purple:
complete target modality x_q
→ frozen EMA Teacher Projector P_q^t
→ normalized stop-gradient target z_q^t.

Connect predicted target paths and z_q^t to Cosine JEPA Loss L_J.
Show a dashed arrow from Student Projectors to EMA Teacher Projectors labeled:
"EMA update after optimizer step".
Clearly mark that the complete target modality enters only the stop-gradient
teacher branch and never enters the auxiliary student/GCNet forward.

At the right side, show the total training objective:
L = L_cls + L_rec + lambda_J L_J.

BAND 3 — INFERENCE (green)
Actual incomplete A/T/V input
→ deterministic availability pattern
→ retained Student Projectors and residual adapters for observed modalities
→ Original pre-graph BiLSTM plus pattern hidden residual
→ one Shared Original GCNet forward
→ Emotion prediction.

Place the EMA Teacher, source-anchored predictor, conditional innovation, and
JEPA loss in a faded crossed-out box labeled:
"Training only — removed at inference".

SHARED-PARAMETER VISUAL ENCODING
- Use identical shapes and a shared-outline enclosure to show that Natural and
  Auxiliary views reuse the same Student Projectors, adapters, pre-graph BiLSTM,
  Temporal GCNet, Speaker GCNet, and classifier-side representation parameters.
- Natural and Auxiliary are two forwards within one training batch, followed by
  one combined backward/update step. Do not draw them as two separate models.
- The teacher projector is an EMA copy of the student projector, not a separate
  supervised encoder and not part of inference.

FINAL OUTPUT
- Produce an editable SVG-style vector figure.
- Use concise module labels inside boxes and place detailed equations in a small
  dedicated predictor panel.
- The diagram must remain readable when scaled to a two-column paper width.
- Before finalizing, verify every arrow against the attached code and remove any
  module not present in the code.
```

## 中文提示词

```text
请作为顶级机器学习论文的科学绘图专家，仔细阅读我上传的三个 PyTorch
模型文件，为已实现的 PLCI-JEPA 绘制一张干净、可编辑、论文级的矢量模型
流程图。只根据模型代码画结构，不要加入代码中不存在的模块。

画布使用横向16:9白底，分成三条水平区域：

第一条为蓝色 Natural classification view：冻结的 A/T/V 特征经过正式自然
缺失mask，只对observed模态执行共享student projector和零初始化residual
adapter，进入Original pre-graph BiLSTM；missing-pattern embedding只加在
pre-graph hidden上，再进入共享Original GCNet。GCNet内部画出并行的Temporal
graph与Speaker graph，每条均为RGCNConv→GraphConv→graph-post
BiLSTM/attention，随后Addition。输出连接Emotion Classifier和Original Linear
Reconstruction，分别得到L_cls和L_rec。标注完整ATV时adapter与pattern residual
bypass，固定参数下恢复Original GCNet计算路径。

第二条为橙色Auxiliary JEPA view：从完整冻结特征的独立副本出发，对每个有效
utterance均匀采样A/T/V/AT/AV/TV之一，只保留observed source；经过同一套共享
student projector、adapter、pre-graph BiLSTM和pattern hidden residual，再对共享
GCNet执行恰好一次auxiliary forward，得到h_J。随后画出source-anchored
predictor的三个部分：source-only base anchor beta_(m→q)、低秩graph-context
correction c_(S→q)(h_J)，以及仅双source使用的有界低秩conditional innovation
Delta_(n|m→q)。双source画两条有方向的预测路径，但二者共享同一个h_J和
context correction。

在其下方用紫色画EMA teacher：完整target modality x_q仅进入EMA teacher
projector P_q^t，得到normalized stop-gradient target z_q^t；它绝不能连接到
student或GCNet输入。prediction和teacher target连接Cosine JEPA loss L_J；从
student projector到teacher画虚线EMA更新箭头，并标注optimizer.step之后更新。
右侧显示总损失L=L_cls+L_rec+lambda_J L_J。

第三条为绿色Inference：实际不完整输入→确定性availability pattern→保留的
student projector/adapter→pre-graph BiLSTM加pattern hidden residual→一次共享
Original GCNet forward→情绪预测。EMA teacher、predictor、innovation和JEPA
loss统一放在淡化删除框中，标注“training only, removed at inference”。

Natural和Auxiliary是同一模型在一个训练batch中的两次forward，不是两个模型；
使用相同外框或shared标记体现参数共享。禁止加入contrastive loss、negative
samples、attention fusion、test-time completion、生成缺失模态、MoE、GAN、KL、
uncertainty head或额外graph layer。

风格要求：平面矢量、圆角矩形、细线、正交箭头、留白充足；Natural用蓝色，
Auxiliary用橙色，EMA teacher用紫色，Inference用绿色，共享模块用深灰；禁止
3D、渐变、装饰性图标和大面积阴影。所有文字必须水平、拼写正确，并确保缩放到
双栏论文宽度后仍然清楚。最终输出可编辑SVG，并在提交前逐条对照代码核验箭头。
```

## 论文图注建议

```text
Overview of PLCI-JEPA. The Natural view follows the official rate-matched
missing-modality protocol and optimizes emotion classification together with
the Original GCNet reconstruction objective. A pattern-balanced auxiliary view
samples one of six incomplete observed subsets and performs one additional
forward pass through the shared GCNet backbone. Source-only base predictions
are refined by a low-rank current-view graph-context correction and, for
dual-source patterns, a bounded conditional innovation. Predictions are aligned
with normalized stop-gradient targets produced by modality-specific EMA teacher
projectors. The teacher and target predictor are used only during training;
inference retains the adapted incomplete-input GCNet path and requires one
backbone forward pass.
```
