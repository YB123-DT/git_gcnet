# OSRAM task delivery agreement

- 权重清理记录（2026-10-22）：用户要求删除V3之前两版被替代实现的checkpoint。
  SourceRelation与12-token SourceRoleTokens合计81个BEST/last/smoke/恢复副本已删，
  13.244GiB，不可恢复；代码、Mask、预测、日志、历史与报告保留，189个证据文件
  SHA256未变。V3/Flat/Nested及其他实验未动，不得把旧版本写成仍可权重恢复。
  精确清单与回执见 `experiments/osram_pre_v3_checkpoint_cleanup_20261022/`。

- 最新补齐授权（2026-10-21）：用户明确V3要三种子，不是仅seed66。保留原seed66
  队列/结果，只新增67/68：MELD2组、IEMOCAP4/6各10组（每seed五折）、MOSEI2组，
  共24组，最终四数据集36个训练作业。MOSI不在本次补齐范围，不增加HPO/方法。
  MOSEI沿用s0/60epochs/1e-3；其余cmerc-repro，MELD50/1e-5，IEMOCAP100/1e-3；
  原Missing、任务、模型、batch和Test-best口径不变。新代码可扩展seed入口，原
  正在运行任务使用旧不可变快照，不被覆盖或重启。
  新24组在独立 `experiments/osram_v3_three_seed_20261021/` 持久接续队列登记；
  旧seed66分类控制器退出后自动接续，避免两个控制器分别计数而突破GPU0=4/
  GPU2=8的总上限。保留全部旧BEST/last/Mask，不终止已有训练，不借用坏GPU4。

- 最新有限夜间队列授权（2026-10-21）：只排 V3，已有 MOSEI seed66继续、不重复；
  新增 MELD seed66一组、IEMOCAP4/6 seed66各五折，共11个新训练作业。
  保留各数据集原环境、学习率/轮数/batch、GCNet random mask和分类逐率Test-best
  ACC选点（UA/WF1来自同次预测）；一阶段训练，仅打开既有V3，不调参/不扩种子。
  biggpu物理GPU0最多4组、GPU2最多8组，按smoke及实际峰值显存/RAM/磁盘准入；
  容量不足排队，不杀别的任务、不改batch、不用坏GPU4。统一持久队列独立目录
  `experiments/osram_v3_classification_20261021/`，不得覆盖旧结果/权重。

- 最新有限训练授权（2026-10-21）：MOSI原Flat seed66，分别固定miss0.0–0.7
  从头训练8个模型，各自只测试同一rate；不是cyclic、不是固定一张Mask，也不是
  Persistent。只修改train_rate_mode/fixed_missing_rate/train_missing_rates。
  原GCNet random mask、主干、数据、s0、batch32/Adam/WD1e-5/MSE保持；每组150轮，
  前100轮3e-4、后50轮3e-5，按同rate Test-best W-F1，保留best/完整last与同次预测。
  对照原cyclic Flat66已接受的同rate exact测试Mask；不启用V3/Nested/新loss。
  独立目录 `experiments/osram_mosi_fixed_rate_seed66_20261021/`，biggpu健康GPU2
  最多4并行、资源不足排队，不改现有V3/Nested任务、不扩种子或HPO。
  最终八率均值属于8个独立模型，不能称为cyclic单模型部署成绩。

- 最新独立 V3 授权（2026-10-21）：实现用户明确指定的 Hierarchical Source–Role
  Interaction：共享512→64坐标、逐来源逐激活Gap配对（343维，不提前平均Gap）、
  同角色跨来源Attention、角色专属Local Query、四个零初始化64→512读值校正。
  只在原Flat输入前校正有效forward Base/Gap，不接旧全局1600维残差。默认关闭；
  不改CHM联合写入、Query、Local Skip或GCNet missing。必要测试/smoke通过后直接
  跑MOSI66（150epochs，3e-4→3e-5）与MOSEI66（60epochs，1e-3），s0/batch32/
  WD1e-5/MSE/原split与逐率Test-best W-F1不变，复用对应Flat exact测试Mask。
  不追加诊断、调参或种子；不影响已启动Nested，不恢复旧Source Relation/12-token。
  独立目录 `experiments/osram_source_role_v3_20261021/`，在biggpu健康GPU1运行。

- 最新独立实验授权（2026-10-21）：用户要求 MOSEI 原版 Nested GNN 三种子与当前
  Flat 比较。使用 `nested_gnn_rooted_evidence`（dim64/depth3/GIN hidden64/groups8），
  不是 root-aware/Local8/直接替换或调参赢家。新 GCNet Missing、s0、60epochs、
  constant1e-3、batch32、WD1e-5及逐率Test-best W-F1保持；Nested66/67/68从头训练。
  当前同配置Flat只有66，复用该结果并补Flat67/68，共5个新作业；不能混入67/68
  的1e-4 Flat或旧ConversationMask的83.843。此任务不恢复已停止的Source Relation
  或12-token训练。独立队列见 `experiments/osram_mosei_nested_gcnet_3seed_20261021/`。

- 最新停止指令（2026-10-21）：用户认为 Source Relation / 12-token 重做并非所需
  设计，要求先停止。新12-token控制器与MOSEI训练已停，MOSI150已完成、MOSEI20
  已保存；所有BEST/last/预测/Mask/日志保留。不得按旧启动授权自动续训、扩种子
  或再补新结构。先核对用户所需计算图，尤其不要把代理补定的12-token全局
  Attention及Flat残差接法当作用户已明确确认。停止证据见
  `experiments/osram_source_role_tokens_20261021/CANCELLED.json`。

- 当前 MOSI 学习率约定（用户要求定下，2026-10-21）：新
  `gcnet-random-mask-v1` 的原 cfg84 no-JEPA Large Flat，150epochs，
  epochs1–100使用3e-4，101–150使用3e-5；其余原设置不变，s0环境不变。
  seeds66/67/68已完成，八率Test-best W-F1均值79.2817，高缺失73.4715；
  同种子原1e-3→1e-4对照为78.8909/72.9079。后续该MOSI训练及模块对照
  沿用此学习率，除非用户另行指定；不改其他数据集或源码全局默认值。
  证据见`experiments/osram_mosi_mid_lr_3seed_20261021/completed/`。
  仍为内部逐率Test-oracle，不混同旧conversation-mask-v2的80.7结果。
  MOSEI本轮1e-4仅新完成67/68两种子，不据此改标三种子baseline或认定LR更优。

- 最新选点硬约束（用户再次确认，2026-10-20）：当前探索实验沿用逐缺失率
  Test-best，MOSI/MOSEI按既定W-F1，IEMOCAP/MELD按既定ACC；UA/W-F1必须
  来自ACC所选同一epoch、同一次预测及exact Mask。不得擅自改为validation
  单checkpoint选点，也不得额外扣除原训练对话做adapter holdout。若用户
  明确指定不同协议再单独记录。全部明确标记内部Test-oracle探索；八率
  可来自不同epoch，不描述为单模型部署成绩。每轮记录逐率测试分数，保存
  八个BEST、匹配预测/Mask与完整last恢复状态，避免再次无法补算选点。

- 当前 MELD Flat baseline（用户指定，2026-10-18）：原 cfg84 no-JEPA
  Flat1600 + Local Skip，Adam lr1e-5、weight_decay1e-5、constant LR，
  50epochs、batch32、gcnet-random-mask-v1、seeds66/67/68。八率 Test-best
  ACC61.557、UA37.728、W-F1 58.509；UA/W-F1 必须来自 ACC 选中的同次预测。
  权重与 exact mask/预测 SHA256、逐种子/逐率结果及完整配置见
  `experiments/osram_meld_wd_seed66_20261018/BASELINE.md` 与 `BASELINE.json`。
  本基线24个BEST和三个last_training必须保留，不清理、不覆盖旧基线权重。
  B1/B2旧实验使用wd2e-4，不得改标为本新baseline上的配对实验；其他数据集不变。
  本登记不授权新增训练，不改变源码默认值；继续明确标记Test-oracle内部探索。

- 历史 MELD Flat baseline（用户指定，2026-10-14）：采用原代码复跑
  seeds66/67/68 的八率 Test-best ACC 平均60.693，对应 UA36.255/W-F1 57.045。
  GCNet missing、50epoch、原1600 Flat 与 Local Skip 均保留；不修改其他数据集
  baseline。完整权重/mask索引与重现边界见
  `experiments/osram_meld_testbest_ua_20261014/BASELINE.md`。
  该基线24个checkpoint必须保留，不纳入旧调参权重清理；测试重放必须复用
  选点当轮mask，不能混用最终复测mask。冻结推理已验证，从头重训逐位一致
  尚未证明；best权重缺少optimizer/RNG，不称为完整断点续训。

- 最新范围覆盖（2026-10-05）：用户明确提供并要求实施 C01–C20 完整机制。
  这是新的有限批次，不是继续扩充/恢复旧120调研队列；允许按各项机制改变
  Memory、表征、任务分布或必要来源损失，不得伪装成同一个 readout residual。
  首轮 MOSI seed66，保留 cfg84 数据/缺失率/选点记录；每项协议差异明列。
  C17 需 binary control，C20 需同任务 mean-gradient control。先完成小规模
  接口/机制检查再部署；正式运行仍在 biggpu，健康GPU2/6，GPU4禁止。
  高开销候选独占，其他候选按实际余量保守并行，不用旧11组显存估计硬塞。
  本批可用 bounded native subagents 实现/复核；旧调研子agent不重启。
  历史连续搜索 goal 保持 paused；不以本批授权自动恢复无限搜索。
- 2026-10-05 后续资源覆盖：C01–C20 只用 biggpu 物理GPU6，禁止启动到GPU2；
  用户要求GPU6并发11组，按真实批次峰值加余量准入，最高11；高峰值方法
  必须避免与其余11组同堆造成OOM。不改变batch或算法维度来强塞。

- 最新恢复授权（2026-10-04）：已写好的M01–M40继续启动，biggpu物理GPU2/6
  每卡最多11组（含原运行），不杀原任务、不增加GPU smoke，空位自动补齐。
  同时子agent扩充120候选池；后续每40项来源核对、机制去重和代码实现完成后
  自动提交/部署/排队，不需逐批再次询问。未完成实现不算可运行实验。
  显式要求的历史同机制配置作为comparison记录，不算新的独立方法。
  此条撤销下一条“暂不启动”的暂停，仍禁止无限扩大120目标或改变训练协议。

- 最新用户覆盖（2026-10-04）：先写代码，暂不启动任何新实验；优先实现用户
  指定的 M01–M40，再继续120模块候选目标。已运行训练保留，等待补位暂停。
  历史同机制只标记复用/配置对照，不因改名或变宽度重复计为新方法。
  此条覆盖下文先启动及资源空闲后自动补位要求，直到用户重新明确启动。

- 用户后续覆盖（2026-10-04）：GPU2/GPU6 各11组并行；本次在已预留58个
  方法外明确增加GPU6的11个不同已实现方法，再允许GPU2补位1个未跑方法，
  总训练预算上限暂为70（2026-10-04后续补位授权）。
  此为本批有限覆盖，不是无限搜索授权；后续40个调研候选不自动扩大训练预算。
  用户要求本批先启动、不做新增smoke；记录CUDA smoke deferred，不伪造通过。

- 用户最新限制（2026-10-03）：meaningful-method 筛选最多 60 个不同方法，
  每轮 20 个、最多 3 轮；替代此前无限循环授权。多种子复验不算新方法，
  重跑不增加方法数；不能用改名或宽深变体扩充名额。达到上限后停止新增
  方法实验，完成已授权复验与汇总，如实报告是否提升，不自动开启第 4 轮。
- 这些方法共用同一套最小检验模板，通过一个就启动一个；不新增通用审查
  轮次。仅在具体报错、非有限值、mask 泄漏或协议不一致时定向排查。
- 每轮更新 `experiments/osram_method_registry.json` 并与全部历史轮次及本轮
  候选做核心机制去重，不只比较方法名。必须记录最接近的历史方法和实质
  差异；改名、换论文包装、改插入位置或仅改超参数均不算新方法。不确定
  的条目保持 pending，不凑名额、不启动。

- 用户要求：每个任务完成并验证后，都必须将该任务相关的代码、实验配置、
  结果或分析记录提交到 Git，不等用户再次提醒。分析任务也应保存可追溯记录。
- 按任务分别提交，提交信息遵循上级 AGENTS.md 的 Lore 协议；报告 commit ID。
- 只暂存本任务文件，保留其他未提交修改；不提交 checkpoint、数据缓存、凭据。
- 用户已明确要求本项目每个任务完成并验证后自动 commit 并 push，不再等待
  逐次提醒。推送到 github remote（YB123-DT/git_gcnet）的当前工作分支，
  不推 origin 上游仓库，不强推。失败时报告原因，不把本地提交说成已上传。
- 没有文件变更的纯问答不制造空提交；不将无关未提交改动混入推送。
