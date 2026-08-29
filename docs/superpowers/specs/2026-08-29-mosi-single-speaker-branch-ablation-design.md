# CMU-MOSI 单说话人 Graph Branch 消融设计

## 目标

诊断 GCNet 在 CMU-MOSI 上是否因单说话人条件而受到 Speaker graph 冗余或干扰。实验只改变 Temporal/Speaker 两个 graph branch 的启用状态，不改变 Missing-M3、Regression-MSE、mask、feature、训练预算或模型选择协议。

## 现有机制

GCNet 对同一个 pre-graph recurrent representation 分别构建 Temporal graph 和 Speaker graph：

```text
pre-graph hidden
├─ Temporal graph（past/current/future 三种关系）→ h_t
└─ Speaker graph（n_speakers² 种关系）          → h_s

Original: h = h_t + h_s
```

CMU-MOSI 的 `n_speakers=1`，因此 Speaker graph 只有一种 `00` relation；它仍使用与 Temporal graph 相同的窗口邻接，但不再携带多说话人交互信息。

## 备选实现与选择

1. **保留两套参数，只路由被选择的 forward（采用）**：参数量、state-dict key、初始化 RNG 与 Original 相同；消融只回答分支计算是否有用。
2. 删除未使用分支：节省参数，但同时改变参数预算、初始化过程和 checkpoint key，归因不干净。
3. 两个分支都执行后将一个乘零：数学上可诊断，但浪费图计算和显存。

## 模式定义

新增：

```text
graph_branch_mode = both | temporal-only | speaker-only
```

- `both`：执行两个分支并相加，必须与修改前逐位等价；
- `temporal-only`：仅构建并执行 Temporal graph，返回 `h_t`；
- `speaker-only`：仅构建并执行 Speaker graph，返回 `h_s`。

所有模式仍实例化 `graph_net_temporal` 和 `graph_net_speaker`。未选择分支不执行 forward，也不产生梯度。分类头、Missing-Latent Predictor 与 EMA teacher 接收所选 graph hidden，不增加额外缩放、融合或补偿参数。

## 接口与兼容性

- `gcnet_modality_jepa.model.GraphModel` 构造函数末尾增加默认 `graph_branch_mode="both"`；
- `gcnet_missing_m3.model.MissingM3GraphModel` 末尾透传同名参数；
- `TrainConfig` 末尾追加字段；CLI 增加 `--graph-branch-mode`；
- config/checkpoint 自动记录该字段；
- 默认 `both` 的参数数量、state-dict key、初始化 RNG 和 forward 输出必须不变。

## 必需测试

1. 非法 mode 抛出 `ValueError`；
2. 默认值与显式 `both` 完全等价；
3. 相同 state dict 下，`both == temporal-only + speaker-only`；
4. temporal-only 不调用 Speaker graph，speaker-only 不调用 Temporal graph；
5. 三种模式参数数量与 state-dict key 完全一致；
6. CLI 到 `TrainConfig` 的透传有 mutation coverage；
7. 三种模式 forward/backward 有限，所选分支有梯度、未选分支无梯度；
8. 原 MOSI regression artifact、IEMOCAP 与 binary task mode 行为不改变。

## 正式实验

- Dataset：CMU-MOSI；Regression-MSE；seeds 66--70；fold 1；
- Frozen features：wav2vec-large-c-UTT、DeBERTa-large-4-UTT、MANet-UTT；
- Slot fusion、hidden 200、window 2/2、100 epochs、`train_rate_mode=all`；
- 一个训练模型用 validation 八-rate 均值选择 checkpoint，并测试 0.0--0.7；
- Original `both` 五种子结果直接继承，不重跑；
- 新运行 5 个 temporal-only 与 5 个 speaker-only，共 10 个任务；避开 GPU4。

## 判读

- Temporal-only 稳定超过 both：Speaker branch 在单说话人 MOSI 上存在冗余/干扰；下一步才考虑条件化分支路由。
- Speaker-only 接近或超过 temporal-only：单 relation graph 仍提供有效局部上下文，问题更可能来自两个分支直接相加。
- 两个 single branch 都低于 both：双分支互补，关闭 Speaker 路线；不再围绕单说话人假设扩展模块。

该实验是结构诊断，不声称新的模型贡献。
