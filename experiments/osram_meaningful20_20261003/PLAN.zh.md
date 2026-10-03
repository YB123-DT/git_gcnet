# 20 个有实质机制的 Local/Base/Gap 模块：持续实验计划

2026-10-03 规划草案；待 Architect → Critic 顺序审查。用户已经授权连续执行，审查通过后直接实施，不再次询问是否继续。完整要求见 [PRD](../../.omx/plans/prd-osram-meaningful20-loop.md) 和 [测试规范](../../.omx/plans/test-spec-osram-meaningful20-loop.md)。

## 目标和边界

每轮收集并接受 20 个不同的非 MSA/MERC 完整核心模块，实现后在 biggpu 上按现有协议训练 seed 66。没有达到预先固定的三种子提升条件就自动开始新一轮 20 个，不用用户反复下指令。逐轮保留拒绝、失败、负结果和重复性核查记录。

Local 是一个 256 维向量；Base/Gap 前向 512 维来自真实的 8×64 读头。只使用当前已有、已经过消融开关处理的读出，不增加 Memory 查询/写入/缓存、不输入标签、不加辅助损失、completion、paired view 或持续调度。保留原 Flat、原任务头及原训练协议。

采用独立 `--osram-meaningful-block=none` 入口、共享安全 residual 边界和按机制族分文件的实现。新增分支经过零初始化输出投影，加到原 Flat 的归一化前 anchor。默认关闭路径、初始化 RNG、dropout RNG、Memory 调用和原参数都须通过回归核查。旧 `osram_readout_candidate` 队列仍保持撤销，不重新启用。

## 固定筛选规则

1. 每轮第一次训练前，20 个不同设计必须全部完成论文/作者代码和可运行映射审查。不能用被拒绝论文、重复 pooling/attention 名称、简单算子或改宽度/深度凑数。
2. 上述 20 个设计接受后，某候选只要自身实现和 CPU/CUDA 检查通过即可启动；不用等全部 20 个写完代码。
3. 完成全部 20 个可比较的 seed-66 结果后再晋级。失败不算负结果，先修复；确实不可行时记录拒绝并以经过同等审查的新设计替换，保留原尝试。
4. 主指标是 0.0–0.7 八档 WF1 的等权均值。Flat seed66 精确值为 **0.8106809539495711**；高缺失 0.5/0.6/0.7 均值 **0.7635225088637502**。以源 JSON 精度比较，不能用四舍五入值。
5. 只晋级主指标严格高于同 seed Flat 的方法；按主指标从高到低取最多 3 个，完全同分按固定 ID 升序。先固定名单，再各补 seed67/68；seed66 复用，负值或持平方法不扩种子。
6. 对照必须具有匹配协议的 Flat seed66/67/68；先核验已有结果，缺失才在 biggpu 补跑。成功条件是三个 seed 的八档均值差再平均后 **严格大于 0**。高缺失和每个 seed 的负值也照实报告，不能事后改指标。
7. 完成固定晋级名单；没有三种子正提升则自动开始下一批全新 20 个。正提升后完成记录、独立核查和任务专属 commit/push。

所有结果都标为内部诊断：per-rate BEST 使用测试集选优，持续架构筛选还会增加选择偏差。三种子正均值不等于显著性或正式论文结论。

## 实施与验证

主集成者独占配置/model/OSRAM/公共边界；独立 executor 按 graph、hypergraph、grouping、set/context、optimization、feature-reasoning 族分文件实现；另一个执行者维护新 runner/queue/summary。不改无关的 `gcnet/model.py`、用户目录或历史结果。所有候选共享固定协议，原始 config 除新增 block 字段外完全一致；补 seed 时只允许登记的 seed 差异。

每个候选检查：真实核心机制、七种 availability、NaN/Inf 污染的 inactive/padding、首个有效话语跳过、future/conversation 独立性、原 Memory 调用数、严格状态加载、有限梯度。零输出第一步内层梯度为零可以是正常现象；先单独测 core，再检查输出层更新后的后续步骤确实更新内层参数。NODE 只能在正常训练 batch 的有效历史行初始化；Hamburger 因子不能跨话语或评测持久更新。

Slot Attention 的训练 epsilon 只由实验 seed 派生的专用 CPU generator 生成，状态持久保存为模型 buffer 并纳入完整续训；不推进全局 CPU/CUDA RNG。评测复用独立 seed1729 生成的固定 epsilon buffer，不改变任何 RNG 或模型状态。

CPU 与真实协议形状的 CUDA 检查通过才入队。记录参数数、训练/评测峰值显存和吞吐。每次执行使用不可变源码快照，保存 commit/差异、源码/config/design/data/environment hash、seed、命令、GPU UUID、PID/启动时间/boot ID、日志和输出目录。后续编码不能污染已运行实验。

## biggpu、并发与恢复

每次 smoke/训练启动前重新核对 GPU index/UUID、进程、free MiB、利用率和磁盘。宿主机 **GPU4 及其 UUID 永久禁止**；`cuda:0` 不能当宿主机编号。0–3 较轻、5 约 90%、6/7 已满只是旧观察，最终用实时状态。

优先在一张健康卡上逐步并发。先完成首轮训练/评测峰值测量；新作业需具备所需峰值再加 `max(2 GiB, 20%峰值)` 余量，磁盘需满足预计产物量再留至少 16 GiB。若算力饱和或总吞吐下降超过 10%，降低后续并发或排队；不更改 batch32/100epochs 等协议。资源不足是等待状态，不擅自跨服务器或删除产物。

队列使用单协调锁、原子状态、启动意图和进程身份，SSH 断开/重启先接管现有进程而不是重复提交。错误分类并保留失败尝试，同一未解决错误不连续重跑。

现有 BEST 只有模型与选优信息，不能完整续训。新流程增加可选独立 last-training checkpoint，保存模型/优化器/进度、适用的 scheduler/scaler、所有 RNG、采样/缺失率状态、历史 BEST 跟踪和协议 hash；用不中断对照验证恢复。没有兼容完整状态时，从头新建尝试，绝不把加载 BEST 权重叫续训。

最终交付保存所有比较和失败项，按 Lore 协议提交，只向 `github` 当前分支推送任务文件，不包含 checkpoint、缓存或无关改动。
