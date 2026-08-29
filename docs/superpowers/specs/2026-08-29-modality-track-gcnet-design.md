# Modality-Track GCNet 设计规格

## 1. 目标与单变量

本实验只改变跨模态融合发生的位置：Control 在图前执行 Slot Fusion；Treatment 让三个
模态保持独立 track，通过共享 GCNet 后再执行同构 Slot Fusion。Missing-M3 Predictor、
EMA、loss、mask、all-rates protocol、classifier 和图拓扑均不改变。

## 2. Track 输入

对模态 `m`：

\[
s_i^m=P_m(x_i^m),\qquad
t_i^m=a_i^m(s_i^m+e_m).
\]

missing 与 padding track 输入严格为零。Projector 只读取 observed block；修改缺失 block
数值不得改变任何 track。

## 3. 共享图推理

三条 track 顺序调用同一个 `encode_hidden`：

\[
h^A=G_\theta(t^A),\quad h^T=G_\theta(t^T),\quad h^V=G_\theta(t^V).
\]

`G_θ` 包含当前共享的 pre-graph recurrent、Temporal graph、Speaker graph、post-graph
recurrent 与 branch addition。三个调用共享参数，不实例化三个 backbone。

## 4. 图后融合

只保留当前 utterance observed track：

\[
f_i=[a_i^Ah_i^A;a_i^Th_i^T;a_i^Vh_i^V;e_{S_i}].
\]

\[
h_i=\operatorname{Dropout}(\operatorname{GELU}(W_f\operatorname{LN}(f_i))).
\]

padding 输出为零。pattern ID 继续使用三位 availability；没有 reliability attention、生成
模态或额外 residual。

## 5. JEPA 与推理

Training-only Predictor 使用原 Student latents 和融合后的 `h_i`，保持当前 loss 不变。
测试只执行三 track GCNet、图后融合和 Emotion Head；Predictor 与 Teacher 均删除。

## 6. 开关与兼容性

新增 `representation_type = slot | track`，默认 `slot`，旧 state-dict/RNG/输出不变。
`track` 与 `raw-residual`、local-context、classification-completion 互斥，正式实验固定
DualGate training-only Predictor。

## 7. 验证与实验

测试覆盖七 pattern、缺失值泄漏、padding、共享参数调用三次、单 track/双 track 融合、
default key 等价、backward、CPU/GPU。正式运行 CMU-MOSI seeds 66--70，每个 checkpoint
测试八个 rates，Control 直接继承。

通过标准：八-rate mean 与 0.4--0.7 mean 均为正，至少 3/5 high-missing seed 为正，
miss0 不下降超过 0.3。计算成本与显存同时记录。

