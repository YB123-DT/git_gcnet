# CMU-MOSI Modality-Slot Fusion

## 研究问题

当前 Missing-M3 使用 observed modality latent 的等权均值构造 GCNet node。该操作可能在进入 GCNet 前稀释 CMU-MOSI 的强 Text 信息。本实验只将其替换为固定 A/T/V 槽位拼接，判断保留模态身份是否改善 MOSI。

## 唯一变量

- Control（继承，不重跑）：`fusion_type=mean`；
- Treatment：`fusion_type=slot`；
- Slot 公式：`[A_slot; T_slot; V_slot; pattern_embedding] -> LayerNorm -> Linear -> GELU -> Dropout`；
- 缺失槽位严格为零；
- M3 Predictor、EMA、JEPA loss、GCNet、mixed-rate schedule 与 checkpoint selection 均保持不变。

本轮不研究六方向梯度冲突，不运行 attention、`lambda_J=0` 或 Original。

## 协议

- Dataset：CMU-MOSI official split，fold 1；
- Features：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`；
- Seeds：66、67、68、69、70；
- 每个 seed 训练一个模型，batch 均衡轮换 missing rates 0.0–0.7；
- 八 rate validation W-F1 等权均值选择一个 checkpoint；
- 同一 checkpoint 测试全部八个 rates；
- GPU：0、1、2、3、5；GPU 4 不使用；
- Epochs：100。

## 实现验证

- 新测试经历预期红灯：4 tests 因 `fusion_type` 参数不存在而失败；
- 实现后：`18 passed`；
- MOSI GPU smoke：1 epoch 完成，train W-F1=0.5242、val8 W-F1=0.2544，forward/backward/EMA/checkpoint/test 均完成；
- 远程 Python：`/data2/yb/reproduction_envs/gcnet-official/bin/python`；
- 远程源码：`/data2/yb/paper/GCNet_TPAMI_single_view_dev`；
- 远程结果：`/data2/yb/remote_experiments/missing_m3_mosi_slot_20260828`。

## 运行状态

正式五种子尚未汇总。训练完成后，本节将记录每个 seed、每个 rate 的 W-F1/MAE/correlation、mean±SD、相对 mean fusion 的配对 delta、正向 seed 数和最佳 epoch。

