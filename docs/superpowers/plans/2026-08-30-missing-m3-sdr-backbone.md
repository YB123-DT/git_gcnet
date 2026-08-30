# Missing-M3 SDR-GNN 整主干替换实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> superpowers-zh:subagent-driven-development 逐任务实现本计划。每个任务先做规格审查，
> 再做代码质量审查；步骤使用复选框跟踪。

**目标：** 在独立目录中实现 sdr-public 与 sdr-paper 两个 SDR-GNN 对话主干，
只替换 Slot Missing-M3 的 GCNet encode_hidden 路径，并完成 CMU-MOSI 五种子、
八 missing-rate 的锁定对照实验。

**架构：** 现有 Slot Observed-Set Encoder 继续把官方不完整输入编码为
[L,B,256]。共享双向 GRU 得到 [L,B,400]，随后由确定性 temporal/speaker
关系图、RGCN、HypergraphConv、frequency-aware highConv 和图后双向 GRU 得到
SDR branch hidden [L,B,500]。sdr-public 只返回 temporal branch；sdr-paper
拼接 temporal/speaker 并投影回 500。Missing-M3 predictor、EMA teacher、
MSE+JEPA loss、分类头和 mixed-rate 训练协议不改。

**技术栈：** Python、PyTorch 1.8、PyTorch Geometric、pytest、现有
GCNet/Missing-M3 训练器、biggpu V100、Git/GitHub。

---

## 文件结构

**创建：**

- gcnet_missing_m3_sdr_backbone/__init__.py：公开两个 variant 与模型。
- gcnet_missing_m3_sdr_backbone/layers.py：确定性 graphify、hypergraph、
  high-frequency convolution 与 SDR branch。
- gcnet_missing_m3_sdr_backbone/model.py：SDRConversationBackbone 与
  MissingM3SDRModel。
- gcnet_missing_m3_sdr_backbone/train_gcnet.py：复用 Missing-M3 生命周期的
  锁定训练入口。
- gcnet_missing_m3_sdr_backbone/run_mosi.py：10 个 treatment job、resume、
  manifest 与汇总。
- gcnet_missing_m3_sdr_backbone/README.md、STATUS.md：边界与状态。
- gcnet_missing_m3_sdr_backbone/tests/：模型、训练器、runner 测试。
- gcnet_missing_m3_sdr_backbone/results/：仅保存 compact summary/provenance。

**不修改：**

- gcnet_missing_m3/model.py
- gcnet_missing_m3/loss.py
- gcnet_missing_m3/mixed_rate.py
- gcnet_missing_m3/train_gcnet.py

## 任务 1：用失败测试锁定 SDR 图语义

**文件：**

- 创建：gcnet_missing_m3_sdr_backbone/__init__.py
- 创建：gcnet_missing_m3_sdr_backbone/tests/__init__.py
- 创建：gcnet_missing_m3_sdr_backbone/tests/test_layers.py

- [ ] **步骤 1：创建最小包入口和失败测试**

测试先导入尚不存在的 layers 符号，预期 collection 因缺少模块失败。直接使用已经
记录的远程 Python，不重复发现环境：

~~~bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest \
  gcnet_missing_m3_sdr_backbone/tests/test_layers.py -q
~~~

- [ ] **步骤 2：锁定 deterministic edge window**

对 lengths=(3,2)、window=(1,1) 断言边顺序是按 conversation、source node、
target node 的词典序，且不同 conversation 无跨图边。temporal relation ID 固定：

~~~python
TEMPORAL_RELATIONS = {"past": 0, "now": 1, "future": 2}
~~~

关系方向保持 public 语义：target index 大于 source index 标为 past，相等为 now，
否则为 future。speaker relation固定按 source/destination speaker 组合的显式表，
不能枚举 set。

- [ ] **步骤 3：锁定 node-order 与 padding**

把 [L,B,D] 的有效前缀 flatten 为 conversation-major [sum(lengths),D]，再恢复到
[L,B,D]。改变 padding 值或 padding qmask 不得改变 graph、有效 node 或恢复结果；
恢复后的 padding 必须严格为零。

- [ ] **步骤 4：锁定 HypergraphConv 与 highConv 数学合同**

使用微型图手算 incidence normalization 和 degree normalization，检查 forward、
finite backward、无边图与单节点自环。highConv 的消息固定为：

~~~python
degree_norm * source_feature * tanh(
    gate(concat(target_feature, source_feature))
)
~~~

不增加新的 linear transform、attention 或 residual。

- [ ] **步骤 5：实现最小图层并使测试转绿**

实现：

~~~python
graphify(values, qmask, lengths, window_past, window_future, graph_type)
conversation_to_nodes(values, lengths)
nodes_to_conversation(nodes, lengths, total_length)
SDRHypergraphConv
FrequencyAwareConv
~~~

优先复用 torch_scatter/PyG 基础操作，但不得复制无许可证上游的大段源码。

- [ ] **步骤 6：提交图语义原子变更**

提交包含 Lore trailers，Tested 记录目标 pytest。

## 任务 2：TDD 实现共享 SDR branch 与两个完整 backbone

**文件：**

- 修改：gcnet_missing_m3_sdr_backbone/layers.py
- 创建：gcnet_missing_m3_sdr_backbone/model.py
- 创建：gcnet_missing_m3_sdr_backbone/tests/test_model.py

- [ ] **步骤 1：先写 shape、padding 与 variant 失败测试**

小模型输入 [4,2,6]，输出维度 11；正式回归输入 [7,3,256]，输出必须
[7,3,500]。sdr-public 注册 temporal branch 但不得注册 speaker branch/fusion；
sdr-paper 必须注册两支和 fusion。

- [ ] **步骤 2：先写双分支梯度测试**

将 paper fusion 权重设置为明确非零，执行 loss.backward 后，temporal 和 speaker
RGCN、hypergraph、highConv、postgraph GRU 都必须有 finite nonzero gradient。
public variant 不能产生任何 speaker 参数。

- [ ] **步骤 3：实现 SDRRelationBranch**

~~~python
nodes = rgcn(node_features, edge_index, edge_type)
nodes = hypergraph(nodes, edge_index)
nodes = high_conv(nodes, edge_index)
conversation = nodes_to_conversation(
    cat([node_features, nodes], dim=-1), lengths, total_length
)
fused, _ = post_graph_bigru(conversation)
hidden = relu(output_linear(fused))
hidden = hidden * valid_mask
~~~

branch 输入宽 400、graph_hidden 100、输出宽 500。time_attn 保持 False，因为当前
锁定 Missing-M3 配置就是 False；不能额外引入 MatchingAttention。

- [ ] **步骤 4：实现 SDRConversationBackbone**

构造前拒绝 bool 形式 dropout，避免 True/False 被静默转换成 1.0/0.0：

~~~python
if isinstance(dropout, bool):
    raise TypeError("dropout must be a real probability, not bool")
~~~

共享 pre_graph_bigru 为 2 层 bidirectional GRU(256,200)，对 padding 输入先置零。
两种 variant：

~~~text
sdr-public: pre-GRU -> temporal branch -> [L,B,500]
sdr-paper : pre-GRU -> temporal + speaker -> concat -> Linear -> ReLU
~~~

输出 padding 严格置零，float32 CPU/GPU forward/backward 均 finite。

- [ ] **步骤 5：实现 public-code-effective parity fixture**

测试从 SDR 官方仓库的 public GraphNetwork 复制 RGCN、HypergraphConv、highConv、
post-GRU 和 linear 权重；按照显式 relation table 对 RGCN relation weight 重排。
对同一 synthetic conversation，在 dropout=0、time_attn=False 下比较 temporal
branch，最大误差小于 1e-6。该测试是开发验证，可在官方源码缺失时 pytest skip，
但 biggpu 正式验证必须实际通过。

- [ ] **步骤 6：使模型测试转绿并提交**

运行 test_layers.py 与 test_model.py；提交 Lore 记录 parity、padding、双支梯度。

## 任务 3：接入 Missing-M3 且不保留旧 GCNet 参数

**文件：**

- 修改：gcnet_missing_m3_sdr_backbone/model.py
- 修改：gcnet_missing_m3_sdr_backbone/tests/test_model.py

- [ ] **步骤 1：先写集成失败测试**

候选必须保留父类 forward tuple：

~~~text
(logits, classification_hidden, latents, predictions)
~~~

hidden 宽固定 500，missing_predictor 和 smax_fc 接口不变。候选 state dict 不得含：

~~~text
lstm.
gru.
graph_net_temporal.
graph_net_speaker.
~~~

evaluation 的 predict_missing=False 时不能调用 missing_predictor 或 teacher。

- [ ] **步骤 2：实现 MissingM3SDRModel**

先执行 MissingM3GraphModel 初始化以保持共享模块初始化顺序，然后删除继承的旧
conversation modules，创建 conversation_backbone。encode_hidden 只做：

~~~python
return self.conversation_backbone(
    self._feature_tensor(inputfeats),
    qmask,
    umask,
    seq_lengths,
)
~~~

pre_graph_residual 非 None 时明确拒绝。本实验只允许 representation_type=slot、
fusion_type=slot、base_model=GRU。

- [ ] **步骤 3：锁定共享参数与初始化**

相同 seed 构造 control/candidate，observed_set、teacher、missing_predictor、
smax_fc 的对应 tensor 必须精确一致。删除旧主干后不得保留 dead parameters。

- [ ] **步骤 4：记录两个 variant 参数量**

参数测试计算 conversation_backbone registered/trainable 参数，并记录整个模型参数；
不把预计值静默硬编码为通过条件，第一次 verified run 产出后再把实际整数写入
runner 的 semantic completion check。

- [ ] **步骤 5：运行集成测试并提交**

同时验证七种 availability pattern、predict_missing=True/False、EMA update、
CPU/GPU FP32 backward 和 padding。全部通过后提交。

## 任务 4：实现锁定训练入口与 10-job runner

**文件：**

- 创建：gcnet_missing_m3_sdr_backbone/train_gcnet.py
- 创建：gcnet_missing_m3_sdr_backbone/run_mosi.py
- 创建：gcnet_missing_m3_sdr_backbone/tests/test_train_gcnet.py
- 创建：gcnet_missing_m3_sdr_backbone/tests/test_runner.py

- [ ] **步骤 1：先写配置与 runner 失败测试**

SDRTrainConfig 只开放 seed、device、epochs、evaluate_test 和 variant。其余固定：

~~~text
dataset=CMUMOSI; fold=1; hidden=200; graph_hidden=100
window_past=2; window_future=2; latent_dim=256
fusion_type=slot; train_rate_mode=all; lr=5e-4
dropout=0.5; task_loss=mse; jepa_weight=0.1; epochs=100
~~~

runner 必须生成恰好 10 个 treatment job：

~~~python
variants = ("sdr-public", "sdr-paper")
seeds = (66, 67, 68, 69, 70)
~~~

命令中不得出现 Original/control。GPU 4 必须被拒绝；默认使用当时健康的 2、3、5、6、7
并支持显式健康 GPU 列表。

- [ ] **步骤 2：薄复用当前训练生命周期**

直接别名复用 gcnet_missing_m3.train_gcnet 的 loader、mask schedule、train_epoch、
evaluate_rate、checkpoint 选择与 metric。只覆盖 build_model 和 metrics provenance。
best checkpoint 唯一依据是 validation 八 rate 平均 W-F1；测试集不得参与选择。

- [ ] **步骤 3：实现原子 resume 与 manifest**

完整结果必须同时满足：

- config 精确匹配；
- history 恰好 100 epochs；
- metrics 记录 validation-selected epoch；
- test 含 0.0--0.7 八 rate；
- 每 rate prediction NPZ 存在并可重新计算；
- mask SHA256 与 metrics 一致；
- variant、source commit、source SHA、config SHA、feature paths、Python/Torch/CUDA
  provenance 完整。

manifest 用临时文件加 os.replace 原子写入；半写文件不能继承。异常子进程必须被
终止并记录，runner 不允许永久等待。

- [ ] **步骤 4：实现 dry-run、resume 和汇总**

dry-run 打印十条命令；resume 只跳过通过语义检查的结果。aggregate 按 variant、
seed、rate 输出 validation/test W-F1、均值、paired delta、positive seed count、
high-missing mean 和 collapse indicators。

- [ ] **步骤 5：测试转绿并提交**

运行：

~~~bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m pytest \
  gcnet_missing_m3_sdr_backbone/tests -q
~~~

## 任务 5：真实 batch 验证与一次短训练

**文件：**

- 创建：gcnet_missing_m3_sdr_backbone/README.md
- 创建：gcnet_missing_m3_sdr_backbone/STATUS.md

- [ ] **步骤 1：同步代码到 biggpu**

使用独立远程目录；正式结果放在独立实验根，避免后续 rsync 删除。同步后记录本地和
远程 source SHA。

- [ ] **步骤 2：运行全测试与真实 MOSI batch**

在官方 Python 下分别对两个 variant 执行真实 loader forward/backward，确认：

- 输入 [L,B,256]，输出 [L,B,500]；
- predictor 收到 500；
- finite loss/gradient；
- GPU 4 未被使用；
- 峰值显存和单 batch 时间有记录。

- [ ] **步骤 3：只做一次 1-epoch 生命周期测试**

每个 variant 各运行一个 seed 的 1 epoch，不保存正式 checkpoint；验证 history、
validation 八 rate、test 跳过逻辑、退出码和 manifest。禁止为每个 seed 重复 smoke。

- [ ] **步骤 4：独立检查并提交文档**

README 明确 paper/public 差异、无许可证重实现、使用命令、保留项、排除项和
Original 继承规则。STATUS 标记 short-run 证据。

## 任务 6：运行正式 2 variants x 5 seeds

**文件：**

- 远程：独立 formal output root
- 更新：gcnet_missing_m3_sdr_backbone/STATUS.md

- [ ] **步骤 1：冻结 source commit 与 manifest**

正式启动前工作树必须干净。manifest 绑定 commit、源码哈希、配置、环境、feature
路径。之后不得根据 test 改代码或超参。

- [ ] **步骤 2：并发启动十个模型**

按健康 GPU 调度，单卡最多按显存测得的安全并发数运行；永不使用 GPU 4。每个模型
训练一个 all-rates-per-batch checkpoint，再测试八个 rate。Original 不重跑。

- [ ] **步骤 3：持续监控而不阻塞实现**

监控 epoch 增长、GPU 利用率、non-finite、退出码、预测标准差和 sign count。异常只
重启失败 treatment，不重跑完整结果；检查时间不做重复环境扫描。

- [ ] **步骤 4：确认 10/10 语义完成**

每个 job 必须有 100 epochs、best validation epoch、八 rate NPZ、metrics 与 SHA。
runner exit code 为零不替代语义检查。

## 任务 7：独立复算、科学判定与上传

**文件：**

- 创建：gcnet_missing_m3_sdr_backbone/results/README.md
- 创建：gcnet_missing_m3_sdr_backbone/results/PROVENANCE.json
- 创建：gcnet_missing_m3_sdr_backbone/results/SUMMARY.json
- 创建：gcnet_missing_m3_sdr_backbone/results/SUMMARY.md
- 更新：gcnet_missing_m3_sdr_backbone/STATUS.md

- [ ] **步骤 1：从预测 NPZ 独立复算 W-F1**

逐 job/rate 重算 W-F1，与 metrics 容差内一致；验证同 seed/rate mask SHA 与继承
GCNet control 配对。若 mask 不一致，标记为非严格配对而不是隐瞒。

- [ ] **步骤 2：执行预注册门槛**

分别报告：

- 五 seed validation 八率均值 paired delta；
- positive seed 数；
- 0.4--0.7 validation mean delta；
- test 八 rate 每 seed/均值；
- representation/prediction collapse 指标；
- 参数量、运行时间和峰值显存。

失败结果也必须完整保留，不以 test 选择 variant 或补调参。

- [ ] **步骤 3：最终验证**

运行全包 pytest、git diff --check、compact result schema 检查和 Git 状态检查。
使用 verification-before-completion 技能，依据实际输出才能声明完成。

- [ ] **步骤 4：Lore commit 与 GitHub push**

不上传 checkpoint、完整 prediction、临时 log 或环境副本；上传代码、测试、
计划、README、provenance 和 compact summaries。推送
feature/missing-m3-sdr-backbone 到 git_gcnet GitHub 仓库。

- [ ] **步骤 5：分支收尾**

使用 superpowers-zh:finishing-a-development-branch；保持分支独立，除非用户之后
明确要求合并。
