# CMU-MOSI Local-Context Residual Fusion

## 研究问题

当前最佳 Slot Missing-M3 先把局部 A/T/V Student slots 压缩成一个 node，再由 GCNet 建模对话上下文。该压缩可能让完整模态条件下的 utterance-local sentiment evidence 在图传播后变弱。本实验只在分类表示处增加零初始化的局部 residual，判断联合学习 local evidence 与 graph context 能否突破 MOSI `miss=0` 的 85.x 平台。

## 唯一变量

- Control（继承，不重跑）：`fusion_type=slot`；
- Treatment：`fusion_type=slot --local-context-residual`；
- Local 输入：三个 Student latents 与三位 availability；
- Local 输出：与 GCNet hidden 同宽的零初始化 residual；
- 分类：`smax_fc(graph_hidden + local_residual)`；
- Missing-M3 predictor 仍只读取 `graph_hidden`。

不增加第二 regression head、gate、attention、loss 或 reconstruction，也不改变 mixed-rate schedule、EMA teacher、M3 predictor、checkpoint selection 和八 rate evaluation。

## 协议与门槛

- Dataset：CMU-MOSI official split，fold 1；
- Features：`wav2vec-large-c-UTT`、`deberta-large-4-UTT`、`manet_UTT`；
- Hidden：200；time-attn：False；epochs：100；
- 第一阶段只运行 seed 66；Slot seed 66 的继承 control 为 miss=0 W-F1 85.69；
- 扩展门槛：seed 66 miss=0 W-F1 ≥87.0；
- 未过门槛不运行 seeds 67–70；通过后五种子正式目标为 miss=0 mean ≥88.0，且 nonzero-rate mean 相对 Slot 不低于 -0.5。

## 实现验证

- TDD 红灯：新增类不存在时，targeted tests 以预期 `ImportError` 失败；
- 第一轮规格审查发现 train-mode RNG 顺序问题：Local dropout 在 predictor 前执行会改变 predictor dropout mask；
- 修复为先从 `graph_hidden` 计算 predictor，再执行 Local residual 和 classifier；新增训练态精确等价回归测试；
- 本地完整测试：30 passed；
- biggpu `s0` 环境：30 passed；
- 官方环境 1-epoch GPU smoke：train W-F1=0.5441、val8 W-F1=0.2544，8 个 prediction NPZ、EMA、checkpoint 均完整；
- smoke `config.json` 与 `best.pt["config"]` 均记录 Local 三项配置；
- 规格审查与代码质量审查最终均 APPROVE。

## 判别结果

seed 66 完成 100 epochs，最佳 checkpoint 由八个 missing rates 的 validation W-F1 均值选择；最佳 epoch=66，val8 W-F1=78.33%。

| Miss | Local-Context W-F1 | Slot control | Delta |
|---:|---:|---:|---:|
| 0.0 | 84.40 | 85.69 | -1.30 |
| 0.1 | 81.81 | 84.30 | -2.49 |
| 0.2 | 76.61 | 80.51 | -3.90 |
| 0.3 | 77.84 | 78.11 | -0.27 |
| 0.4 | 76.52 | 76.39 | +0.13 |
| 0.5 | 73.48 | 74.02 | -0.53 |
| 0.6 | 71.29 | 73.01 | -1.73 |
| 0.7 | 72.25 | 73.20 | -0.95 |

- miss=0 扩展门槛：84.40 < 87.00，FAIL；
- miss=0 相对 Slot：-1.30；
- nonzero-rate mean：75.68，Slot 为 77.08，delta=-1.39；
- 8 rates 中仅 miss=0.4 为正。

因此按预注册规则停止，不运行 seeds 67–70。Local-Context residual 没有解决 MOSI；它反而说明当前问题不能通过在 frozen Student/GCNet classification hidden 末端增加一个小型局部分支解决。结合 Raw-Residual、hidden sweep、time-attn、threshold oracle 和 prediction ensemble 均未达到目标，下一阶段应转 upstream encoder adaptation/LoRA，或重新检查 MOSI 训练目标与 checkpoint protocol，而不是继续向 frozen-feature GCNet 叠加分类融合模块。

## 完整性与 provenance

- 8/8 prediction NPZ 的 accuracy/W-F1 从原始 prediction 与 label 独立重算一致；
- 8/8 test mask SHA256 与同 seed Slot control 完全一致；
- history 为 100 epochs，6,700 个数值字段全部有限；
- parameter count=32,417,407；
- remote checkpoint SHA256：`474a4d910db813590ecb6a7c3af1a04daf45067d1ee7cc48fff8310ccb51120e`；
- checkpoint 仅保留在 biggpu，轻量 config/history/metrics/prediction NPZ 已回传仓库。
