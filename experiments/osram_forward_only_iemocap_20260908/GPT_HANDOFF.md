# 给 GPT 的问题交接：OSRAM 单向 memory 如何改进？

请先分析证据与机制，不直接建议堆模块或启动实验。

## 当前架构

冻结 A/T/V utterance features → observed Student projectors → masked mean node e_t。
Local(e_t) 与 OSRAM memory 并行。Memory 保存可见模态 key/value，按 utterance
先 decay、再 Base/Gap read、最后 permutation-invariant block-delta write。
Base 始终开启；Gap-A/T/V 只在相应模态缺失时启用。当前 e_t 条件化 query。
Local + Base + 三个 Gap 固定槽位融合得到情绪分类 hidden。
MMoE 在训练时利用可见 latent 和 Base/Gap context 预测 EMA target latent；
测试不回灌预测 latent，predictor/teacher 不用于分类推理。

## 这次唯一变化

原版正向扫描 1→L，反向扫描 L→1；预测 t 可读历史和未来。
单向版仅扫描 1→L，t 的 memory value 来自 1…t−1，Local/Query 可使用 t。
跳过反向 scan，反向 context 槽位置零；其余参数形状、训练协议和 loss 不变。
不是逐个历史前缀重做反向扫描，也不是改成另一种新的状态空间架构。

实现位置：gcnet_missing_m3/osram.py 的 bidirectional 开关；
gcnet_missing_m3/model.py、train_gcnet.py 的 osram_bidirectional 配置。
代码 commit c5945df；MOSI 结果 d38e5e0；IEMOCAP 启动 d27b58f。

## 已完成证据（五种子八率均值，W-F1 %）

| Dataset | 双向 | 单向 | 差值 pp |
|---|---:|---:|---:|
| MOSI | 80.78 | 79.72 | -1.06 |
| IEMOCAP-4 | 84.26 | 82.03 | -2.23 |
| IEMOCAP-6 | 64.88 | 63.42 | -1.46 |

三者高缺失下降总体更明显。IEMOCAP-6 miss0: -0.69 pp；miss0.7: -3.06 pp。
IEMOCAP 完整逐率表见同目录 RESULT.md，逐 seed/rate 最优 epoch 见
per_seed_rate.csv，原始配置、100-epoch histories、metrics、diagnostics 在 raw/。
MOSI 见 ../osram_forward_only_mosi_20260908/RESULT.md。

全部为逐 rate 独立 Test-oracle 内部诊断，不是正式泛化证据；
IEMOCAP official validation/test 都为 Session5。不可拿这些选点结果
直接声称优于其他论文。未来不影响当前的单元测试已通过，但不是完整特征
提取流水线的因果审计。

## 需要区分的问题

1. 信息减少：单向不能获得未来证据；这不是模型 bug。
2. 有效容量变化：参数数量没变，但半边 context 槽位为零，融合有效输入容量降低；
   本实验不能把损失全部归因于未来信息本身。
3. 为什么高缺失对历史 state 的要求更高？同模态历史稀疏、记忆覆盖与检索
   是否可能是瓶颈？目前仅是假设，不能凭 F1 确认。
4. 若要求仅看历史，如何保持简单的一次 O(L) scan，而不是 O(L²) 前缀反扫？
5. 是否应接受单向的合理上限差距，而不是强求没有未来也达到离线双向分数？

诊断注意：当前代码中 memory_frobenius_norm 名称不准确，实际统计的是
base-read norm，并非真实 memory 矩阵 Frobenius norm；单向模式还包含零的
反向 read 平均。不要用这一字段作 memory collapse 结论，本轮未修改它。
仅凭这份 F1 表也不能断言没有表示坍塌；需专门表示统计才能判断。

## 希望 GPT 给出的答复

先给上述下降的合理解释，区分已证实与待验证。若必须保持因果，推荐一个
最小、可归因的改进方向，明确替换位置与理由；优先利用现有 memory，
不改上游特征、loss、MMoE，不默认加 attention/Transformer/新分支。
不要将“删除后下降”解释成不能改主干；也不要假设未来信息能无损恢复。
本次只请求设计建议，不授权自动实现或继续调参。
