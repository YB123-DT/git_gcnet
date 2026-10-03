# Local/Base/Gap 二十候选实现计划

INTERNAL DIAGNOSTIC ONLY

**已撤回，禁止启动旧清单。** 用户要求改为有完整机制的较大 Block，而非基础算子清单。以下是已停止的原计划，不是当前执行指令。未部署、未训练；源码保持默认关闭，仅作可追溯记录。

**目标：** 从非 MSA/MERC 原始论文和作者代码筛出二十种不同机制，在原 cfg84 Flat 上逐项单种子训练，不追加性能驱动调参或诊断。

**架构：** 保留 Local Skip、原 Flat Adapter、LayerNorm 和任务头。新增独立可选 `osram_readout_candidate`（默认 `none`），只在 Flat pre-norm anchor 上加入候选 residual。共用安全输入/输出包装，候选内部算子不同。

**技术栈：** 现有 PyTorch、unittest、SSH biggpu；不增加依赖。

## 固定边界

- 基线代码 `5eb6061`；原 seed66 配置和结果来自 `osram_mosi_memory_gap_ablation_20260920/full/seed_66`。
- MOSI seed66，100 epochs，batch32，Adam lr=.001/wd=.00001，原 task MSE、random cyclic missing 0.0–0.7、原测试 masks、逐率 BEST。只报告内部 Test-oracle 筛选，二十选优有选择偏差，不称正式泛化提升。
- 不修改 Memory/Query/Key/Value/写入、ObservedSetEncoder、分类头；不启用旧 Gate、Relation、completion、JEPA、双视图或新损失；一阶段从头训练。
- 统一输入先投影为128维，固定顺序 `[Local,Base,GapA,GapT,GapV]`；只用 causal forward half。Base 对有历史的有效 utterance 生效，Gap 额外要求该模态缺失。首个有效 utterance 的整个新增 residual 为零。
- `torch.where` 在投影前后安全屏蔽 inactive/padding；不使用 batch 内其他对话特征；无新历史缓存。
- 所有候选输出128维，经相同的零初始化 `Linear(128,output_dim)` 接回原 Flat pre-norm；不替换 Flat。新增初始化隔离 RNG，候选不额外使用随机 dropout。只检验这些明确的适配版本，不声称完整复现原论文。
- 记录不同候选真实参数量；共同外壳不等于严格参数匹配。二十个不同机制，不用重复换宽度凑数。
- 允许候选因来源不清/算法不适配而在正式训练前替换；锁定清单后不根据测试分数改候选。

## 步骤与文件责任

- [ ] 三组文献核验：`literature/bilinear.json`、`recalibration.json`、`interactions.json`。每项含作者数组、canonical URL、作者代码与公式、迁移区别和局限。
- [ ] 先写失败测试：`tests/test_readout_candidates.py` 验证 factory、形状、20项、全屏蔽/NaN安全、first-history、前后向、零初始化、有效更新。
- [ ] 实现独立候选文件及包装：`gcnet_missing_m3/readout_candidates.py`；按机制分文件，避免改写现有主干。
- [ ] 集成配置/model/OSRAM/checkpoint builder；`tests/test_readout_candidate_integration.py` 验证默认关闭、共同参数初始化和 RNG、单次 scan、零初始化输出一致、旧 checkpoint 严格加载。
- [ ] 固定运行器 `run.py` 和 `queue.py`；先写测试验证仅候选字段变化、seed66/100epoch锁定、坏卡禁止、独立目录、完成检查与不重复启动。
- [ ] 全部候选 correctness 通过后统计参数，锁定 `manifest.json`；独立源码快照，提交推送。
- [ ] biggpu GPU5 先 smoke 测真实显存/吞吐，再逐步增加至最多5个本项目进程；当前0–3已有外部队列、6–7高占用，不能抢占或终止无关进程。GPU4禁止。固定 batch/精度不变。
- [ ] 每 run 保存配置、代码哈希、环境、PID/UUID、history、per-rate BEST 和预测；保留失败记录。不重新运行已成功配置，不跨服务器。
- [ ] 完成后汇总每率W-F1、八率mean、高缺失mean及相对同seed Flat差值；所有20项、失败和缺失项完整报告，不仅报告获胜者。

验证命令使用本地 `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest`；实际训练使用已有 biggpu `/data2/yb/reproduction_workspace/envs/s0/bin/python`。不复制权重回本地、不提交权重到Git。
