# 草案：缺失条件匹配的TRAIN实例读取

Status: proposed_not_implemented — requires acceptance of inference-time support bank.

## 检验对象

固定原有数据、CHM定义和任务，检验新增的跨TRAIN实例访问是否有价值。
不预设Memory丢失信息、Flat不能用历史或注意力失败。
本草案是TabR式能力迁移，不是新的论文创新证明。

## 数据边界

1. 支持库只含当前数据划分的TRAIN话语、原始特征、真实TRAIN标签和稳定conversation/utterance ID。
2. 当前样本的真实标签不作为Block输入；val/test标签不入库、不参与索引或邻居过滤。
3. TRAIN query排除整个当前conversation，不只是自身，避免同对话标签捷径。IEMOCAP若以后运行，每fold各自建库。
4. 每个支持话语使用与query相同的当前availability，缺失维度安全置零；不借支持样本把query缺失Text内容补出来。
5. 这会增加推理时的存储/计算/训练数据依赖。不是只保存模型参数即可部署的方案，需要先接受此条件。

## 精确计算草案

基线一次正常CHM扫描保留原Local256、Base/Gap及Flat pre-norm anchor1600：`u0`。
候选库话语只经过共享ObservedSetEncoder及同一Local path，不运行支持conversation的CHM。
因此支持的是“其他TRAIN当前观测示例”，不是模拟出来的其他对话历史。

目标和支持Local进入同一个坐标：

`k = Linear256→128(LayerNorm256(local))`

同一query对相同mask、不同conversation的TRAIN支持按`-||k-k_i||²`取最多16个邻居。
索引只依赖特征，不依赖目标标签；softmax仅覆盖合法邻居。少于16时使用实际合法数，无邻居时分支严格为零。

`v_i = Linear1→128(y_i) + T(k-k_i)`

`T = Linear128→128 → ReLU → Linear128→128(bias=False)`

`r = sum_i softmax(-||k-k_i||²) * v_i`

`hidden = original_emotion_norm(u0 + Linear128→1600(r))`

之后使用原task head和原MSE。Local Skip、原Flat、原Base/Gap均不替换。
只对valid话语读取；padding输出严格为零。
分支可在首句工作，因为TRAIN支持不是当前对话过去；不能错误套用“无对话历史则全分支零”的CHM规则。

保留作者的key-only/L2/target-dependent Value机制；增加了mask匹配与整conversation排除，
接回原1600 anchor而不是替换整模型，所以明确称为adaptation，不称原论文全量复现。

## 初始化、梯度与成本

- 新增模块默认关闭，关闭时不建支持库、不增加参数或RNG消耗，旧checkpoint严格路径不变。
- 新增层初始化隔离到fork_rng，原模块初始化不变。
- 草案采用普通非零输出投影，不引入zero-init出口；启用初始预测不保证等于Flat，这一点必须如实记录。
- 新分支不额外加Dropout/Norm；共享Encoder原有Dropout保留。新增支持编码需隔离Torch RNG，不能改变原Missing sampler状态。
- 一阶段joint training，不冻结原主干、不增加辅助loss或独立LR。
- 第一版MOSI用同一次当前参数的支持编码和精确距离搜索；不使用长期陈旧latent库、不装Faiss。
- 不对离散top-k索引求梯度，但被选邻居的距离/Value/共享编码必须有真实梯度。
- 单批可按7个mask分组编码，不运行第二条CHM轨迹；任何缓存需绑定checkpoint和mask，不能沿用其他权重版本。
- 额外参数按拟定尺寸算术估算为272,960，未构建/实测；库是非参数存储，需另报大小、峰值显存、总耗时。

## 有限对照（仅建议，尚未授权启动）

| 条件 | 候选来源 | 邻居Value | 用途 |
|---|---|---|---|
| F0 | 无新增库 | 原Flat | 复用同协议seed66 reference，不混旧80.7结果 |
| F1 | 相同合法TRAIN库 | `Linear(0) + T(k-k_i)` | feature-only检索，对照额外结构/邻居特征 |
| F2 | 相同合法TRAIN库 | `Linear(y_i) + T(k-k_i)` | 检验TRAIN标签条件化实例证据 |

F1/F2同网络、初始化、数据、mask、更新次数和优化器；不是冻结主干实验。
F1标签Linear的128个weight因零输入不参与学习，bias仍可学习；名义参数一致不等于有效自由度完全一致。
若第一轮确有信号，后续才考虑label-shuffle控制，当前不自动追加第三组。

首轮建议仅MOSI66，F1/F2两次训练：原150epochs、前100轮3e-4/后50轮3e-5、Adam、batch32、WD1e-5、原dropout与MSE。
原gcnet-random-mask-v1/cyclic0–.7保持，复用现有同seed exact评估Mask；按既定逐率Test-best W-F1标记内部探索。
原参考见`experiments/osram_mosi_mid_lr_3seed_20261021/`，seed66 W-F1约79.249、高缺失约73.410，正式前从精确产物重核。

## 必须先验证

- 所有邻居ID属于TRAIN，且query当前conversation完全排除。
- 只按特征选邻居；改变val/test标签不影响预测、邻居、库或训练统计。
- query被mask掉的模态、支持被mask掉的模态均不泄漏；7种组合非空。
- CHM主状态只扫描一次；候选编码不写入主状态，不改变原联合BlockWrite。
- disabled前向、共享初始参数及sampler RNG相等；enabled不要求初始预测相等。
- top-k/共享key/label/T/output都存在预期有限梯度；F1 label weight无梯度属已知对照，不算连接成功。
- checkpoint与支持集manifest保存/恢复相符；库跨checkpoint不能静默复用。
- 一个真实MOSI batch的前向、反向及optimizer smoke，报告计算/内存成本；未通过不启动正式训练。

## 停止与解释

F2需同时超过F0和F1，才有继续考察label-conditioned retrieval的依据。
仅单seed微小波动不是成功，第一轮不自动扩seed/数据集/超参。
如果F1/F2近似相同，不能把共同收益归因于TRAIN标签；若都不优于F0，停止本实现，不立刻增加邻居/层数。
即使成功，也不能宣称找到了OSRAM原瓶颈，更不能称训练标签检索本身具有原创性。

## 确认点

首先需要接受“推理携带TRAIN实例及TRAIN标签库”的部署条件。
接受后再确认上述两组MOSI66有限预算；不授权旧队列复活或其他GPU任务变更。
