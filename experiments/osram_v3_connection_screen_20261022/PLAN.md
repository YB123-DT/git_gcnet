# V3三种接法的有限MOSI/MOSEI实现计划

> 执行采用测试先行和独立文件子任务并行；主代理负责集成、规格审查、质量审查与启动验证。

目标：实现用户已批准的三种V3变体，并启动MOSI/MOSEI各seed66，共六组从头联合训练。不是三模块叠加，不追加Flat复跑、种子或HPO。

架构：沿用原JointSourceTracker及CHM。新模块独立文件，默认`standard`保持旧V3完全不变；新增`head-preserving`、`source-slots`、`flat-hidden`三选一。方案编号1/2/3分别对应前文研究记录C/A/B。Local Skip始终原Local。

技术栈：现有PyTorch/Adam，biggpu既有s0环境和GCNet random missing。不得新增依赖。

进度（2026-10-22 17:05 UTC）：步骤1–3完成，91项本地及65项远程测试通过；六类真实GPU smoke全部通过。六组正式任务已在GPU2/7启动，MOSI及MOSEI均已产生正式epoch记录，训练继续由持久controller管理。实际状态与PID见LAUNCH.json。

## 文件和职责

- `gcnet_missing_m3/source_role_v3_connections.py`：三个接法及共享内部计算；不修改旧`source_role_v3.py`。
- `gcnet_missing_m3/osram.py`、`model.py`、`train_gcnet.py`：新增`osram_source_role_v3_variant='standard'`，opt-in配置和最小Flat入口路由。
- `tests/test_source_role_v3_connections.py`：维度、mask、梯度、参数量与保存恢复。
- `tests/test_source_role_v3_connections_integration.py`：默认关闭/原V3等价、Memory一次扫描、初始化/RNG、Flat输入hook及因果性。
- 本目录`worker.py/run.py/report.py`：复用已验证数据/掩码/选点/恢复接口，六组独立输出、持久队列、显存准入、自动汇总。
- `tests/test_v3_connection_screen.py`：六组且仅seed66、配置白名单、学习率与安全调度。

## 步骤1：模块红灯→绿灯

- [x] 先写失败测试，断言新类存在且支持三个variant；运行本地CPU pytest确认缺少实现时失败。
- [x] 实现`V3ConnectionBlock(local_dim, forward_dim, width, pair_hidden, num_memory_heads, flat_hidden_dim, variant)`。forward输入完全沿用V3的六参数，1/2返回`[T,B,4,forward_dim]`，3返回`[T,B,flat_hidden_dim]`。
- [x] 1：head专属64→64坐标投影和role/head decoder，核心跨head共享，head embedding零初始化；标准参数335518。
- [x] 2：Layer2的各角色3来源64固定槽concat192→512，删除末端query/MHA；标准参数514718。
- [x] 3：保留角色query→4×64，去四个decoder；LN256后重新role-mask、无bias普通初始化256→Flat实际首层hidden；标准参数611998。
- [x] 验证激活与非激活NaN隔离、时间/对话独立、参数更新、保存恢复。输出必须hard-mask，尤其新增LN的bias不得复活inactive角色。

核心断言：`sum(p.numel() for p in block.parameters()) == expected[variant]`；零出口1/2初始输出全0；3的输出非零但原Flat末层0保证完整模型初始预测等价，不能宣称已训练Flat加载后的等价。

## 步骤2：主模型红灯→绿灯

- [x] 新variant非standard必须要求V3开启、residual模式及原Flat接法；拒绝与其他候选/消融旁路不兼容的组合，不默默忽略配置。
- [x] 构造新模块使用原fork_rng机制；默认standard继续实例化旧类，旧checkpoint state keys不变。
- [x] 1/2复用原Base/Gap有效forward加法；3只在原Flat首Linear输出、GELU前加新项，原input LN/LocalSkip/head不变。
- [x] 同seed验证新增模块以外state_dict与RNG一致、初始预测等价、无缺失/无历史旁路、未来改动不影响过去、只扫描一次、恢复一致。

运行：使用现有`multimodalerc310` Python/Torch2.2.2并在搜索路径末尾附加现有`msa_extract`的pytest，完整可复现命令见`VALIDATION.md`。原计划的`comp-repro`缺少pytest，`msa_extract`缺少torch_geometric；没有安装或升级依赖。

## 步骤3：实验入口

- [x] 配置严格保留Flat1600、latent256、CHM8×64、write_step0.6、batch32/Adam/WD1e-5/MSE/原split、cyclic0–0.7与test-best W-F1，关闭其他Block/Loss。
- [x] MOSI150轮，前100轮3e-4、后50轮3e-5；MOSEI60轮constant1e-3。沿用研究参照的完整同预算协议，不将旧100轮与150轮混报。
- [x] 复用原Flat seed66 exact masks，逐率保存BEST及对应预测/labels/mask、完整last模型/optimizer/RNG。与原参考逐epoch mask_trace和optimizer_steps核验。
- [x] 六组共享不可变源码，独立run目录；同一variant仅一次完整Memory扫描。记录新增与总参数、源码/数据hash、实际配置、设备UUID、显存和时间。
- [x] 真实训练批次smoke需至少打开双零初始化后的梯度；不是只看首次梯度0就通过。三变体分别校验实际存在的参数组，不能沿用标准V3必须有decoders/query的错误断言。

## 步骤4：部署与启动

- [x] 本地相关测试通过，独立规格/质量复核；Lore提交仅本任务文件，推送github当前分支。
- [x] SSH biggpu检查当前任务、GPU UUID、RAM、磁盘、s0环境、基线reference、数据hash；排除坏GPU4。
- [x] `git archive`部署提交快照到独立远端目录，不同步覆盖已有源码。
- [x] 优先健康空闲GPU2，按smoke峰值加安全余量准入并发；可使用其他健康卡的真实余量，但不杀/暂停原任务、不改batch。
- [x] 持久controller完成smoke后启动六组，资源不足的本任务自动补位。每个worker支持明确完整状态resume，失败如实记录，不静默重试或改参。
- [x] 核验真实训练PID、首轮训练/测试记录及自动汇总，提交启动与验证记录。长训练仍在运行时明确标记running，不伪称完成或提升。

研究边界：三者均未证明有效；所有成绩为内部逐率Test-oracle。新代码不触碰受保护checkpoint，未获授权不扩大方法/种子/超参范围。

完成记录（2026-10-22 22:43 UTC）：六组均已完成150/60轮，队列complete，无报错。48份BEST权重及对应预测保留，预测hash核验通过；完整逐率与均值见RESULT.md/REPORT.json/PER_RATE.csv。没有自动扩种子或启动新实验。
