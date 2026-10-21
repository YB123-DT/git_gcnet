# Source Relation Attention 实现计划

> **面向 AI 代理的工作者：** 使用 superpowers:subagent-driven-development；模块与测试由单独执行者负责，集成和运行由主代理负责。

**目标：** 实现用户指定角色投影/同源关系/SelfAttention/Local CrossAttention，仅初筛MOSI和MOSEI各seed66。

**架构：** 保留原Flat、CHM和来源追踪。新分支只接真实forward来源读值：四个角色专属512→64投影，跨来源共享；active Gap mean；共享关系MLP263→128→128；一层128维4-head SelfAttention；Local256→128作CrossAttention query。输出128→实际output_dim零初始化，加入原Flat pre-norm，再原emotion_norm/task head。

**技术栈：** 原s0/PyTorch2.2.2/NumPy1.24.3；原GCNet random mask；无新依赖。

## 已验证执行状态（2026-10-21）

- [x] 模块与opt-in接口、mask/RNG/legacy回归及真实来源梯度：39项CPU测试通过。
- [x] 独立规格与代码审查；补齐输出层以外梯度覆盖和逐进程资源准入。
- [x] 源码6dbcc986c451688ea33f33d3ffe6785e0be775f5已提交/push；远程9588个Git blob完全匹配。
- [x] MOSI/MOSEI真实GPU smoke：2/3个optimizer steps，25条参数/输入梯度均非零，原/新模块均更新，完整模型/optimizer/RNG恢复通过。
- [x] GPU7两组正式seed66任务并行，MOSI150/MOSEI60；已观察到17/1轮完成且无报错。自动完成汇总入口已挂在持久控制器。
- [ ] 全部正式训练结束、最终逐率分数及与Flat差值报告。当前不把早期分数/Smoke当成绩。

下文为原执行清单；进度证据统一以本节、SMOKE.json、LAUNCH.json及远程SUMMARY为准。

## 范围与默认值

- 不叠加source_role旧gamma、TASR、Nested、JEPA、completion、双视图或新loss。
- opt-in `osram_source_relation=False`；仅plain causal embedded-beta Flat可启用。
- 新分支不加Dropout（原Flat dropout不变），避免改变原模型训练/采样RNG。
- 关系MLP输入严格为[b,g,b*g,abs(b-g),availability,trajectory_s]，263维；轨迹count/age采用log1p，coverage/ever_seen保留。
- 三个128维z在SelfAttention前加3×128来源身份embedding，避免后续attention丢失来源标识。这是明确的身份编码，不增加来源权重网络。
- 不加额外Transformer FFN、辅助loss或证据中心化减法。SelfAttention输出作为CrossAttention K/V；Local投影作Q。
- source never-seen/padding均mask；所有source不可用时跳过attention，不使用all-masked softmax。
- role projection用bias=False，inactive Gap投影后仍置0。没有active Gap时g=0。
- 无历史及padding在最后有bias的输出之后再次置0；inactive source embedding不可泄漏。
- 新模块初始化隔离CPU RNG。关闭时strict legacy state/output/RNG不变；开启初始输出精确等价。
- 初轮仅原任务loss、所有原参数及新模块共同训练；旧来源缩放路径退出，不删除记录。

## Task1：模块与单元测试（执行者所有权）

文件：`gcnet_missing_m3/source_relation.py`、`tests/test_source_relation.py`。

- [ ] 先编写测试：零初始化、mask/NaN安全、首句/padding零输出、无active Gap、有限梯度、多步后上游attention/role-projection非零梯度、保存重载一致。
- [ ] 在远程s0临时隔离目录执行测试，确认缺少新模块的red failure；不修改受保护源码。
- [ ] 实现`SourceRelationAttention(local_dim,forward_dim,output_dim)`，forward(local,source_base,source_gap,trajectory,availability,umask)→residual。
- [ ] 模块保存简单diagnostics：residual norm、source token norm、active source count、cross-attention weights；不赋予可靠性或情绪冲突语义。
- [ ] 单元测试green后报告，不提交其他人的变更，不启动训练。

## Task2：opt-in路由与回归（主代理）

文件：`osram.py`、`model.py`、`train_gcnet.py`、原model factory `experiments/osram_meld_flat_source_combinations_20261013/evaluate.py`。

- [ ] 编写配置透传/初始化/默认路径不变集成测试，观察缺少开关的失败。
- [ ] 新bool沿TrainConfig→model→OSRAM传递；新增CLI开关。
- [ ] 启用时强制source tracking真实计算图，与旧source_role/TASR互斥；额外参数在原初始化完成后fork_rng构造。
- [ ] 最终Flat分支只新增`emotion_norm(flat_anchor + residual)`；原LocalSkip保持原Local。
- [ ] 运行原source-tracker/source-role回归；严格legacy加载、训练态RNG、输出等价；两步梯度检查不把zero-init第一步零上游梯度误判断路。

## Task3：有限实验入口与配置（主代理）

目录：`experiments/osram_source_relation_attention_20261021/`。

- [ ] 复用原MOSI/MOSEI setup/build-mask/evaluate/checkpoint函数；不重写sampler、特征或指标。
- [ ] MOSI150：seed66，lr epochs1–100=3e-4，101–150=3e-5。MOSEI60：seed66，constant1e-3。batch32/Adam/wd1e-5及原dropout保持。
- [ ] 原task MSE；cyclic0.0..0.7；per-rate test WF1 strict-greater earliest tie；同次预测ACC/WF1/exact masks；八BEST+完整last optimizer/RNG保存。
- [ ] 一次真实GPU单batch多步smoke并保存梯度/数值/开销证据；仅此两组训练，不搜索、不扩种子。
- [ ] 固定Git源码快照，保存配置/源码哈希/数据指纹/环境/GPU UUID/运行命令；健康GPU白名单排除4，不杀任何其他任务。
- [ ] 资源允许同卡并行两组；持久tmux运行，连接失败先核实进程避免重复启动。

## Task4：审查、交付与结果

- [ ] 对执行者模块先规格审查，再代码质量审查；修复必要问题。
- [ ] 代码、测试和配置提交/push github当前分支，不推origin上游。
- [ ] 实际训练完成才写分数；运行中仅报告真实epoch/日志/PID。
- [ ] 输出per-rate ACC/WF1、八率和high means，参数数/峰值显存/时间，并与同协议原Flat seed66对照。
- [ ] 结果明确INTERNAL TEST-ORACLE SCREEN，不冒称独立测试泛化、纯模态语义或机制成功。

该用户已明确要求实现并运行，无需再次请求启动授权；本计划不授权额外对照训练或扩种子。
