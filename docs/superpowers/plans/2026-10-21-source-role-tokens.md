# Source–Role Token 重做计划

**目标：** 按用户“重新做”要求，取消进入 Attention 前的 Gap mean，独立保留 4×3 来源—读取角色槽位。旧 mean 版是旧公式的实现，不改标为代码错误，不覆盖其训练证据。

**明确的实现选择：** 12 个角色优先 token（Base、Gap-A、Gap-T、Gap-V，每行 A/T/V）各自经过角色专属 512→64 投影；拼接角色 8 维、来源 8 维、availability 3 维、来源轨迹 4 维，经共享 87→128→128 MLP；一层 128-d/4-head SelfAttention 后，Local 256→128 CrossAttention。没有 mean、sum 或 source-token 压缩出现在 SelfAttention 前。输出仍是零初始化 128→1600 的原 Flat pre-norm 残差，仅更换历史编码，不同时更换最终分类路径。

**不变项：** 真实来源追踪、CHM 读写/Query/KV、Flat/Local Skip、原 task MSE、GCNet random mask/RNG；MOSI66 150 epochs，3e-4 前100轮/3e-5 后50轮；MOSEI66 60 epochs，1e-3；batch32、WD1e-5、s0 环境；逐率 Test-best W-F1，仅内部探索。新实验独立输出，不从旧 mean 权重续训、不扩种子或调参。

## 执行清单

执行状态：64项远程CPU测试通过（另1项Git历史专属测试显式跳过），12项本地runner测试通过；Memory四个核心函数AST未变。源码da4a9f4已push，9600个远程Git blob全部一致。MOSI/MOSEI真实GPU smoke均通过（38/38梯度路径含实际12槽），GPU7持久控制器已启动两组新训练。完整训练结果尚未完成，不据此宣称提分。证据见新实验SMOKE.json/LAUNCH.json。

- [ ] 测试优先：`tests/test_source_role_tokens.py` 断言 Attention 真实接收 12 个 token、逐槽内容/身份、inactive/padding/never-seen NaN 安全、零初始化、源/角色梯度。远程 s0 先运行确认新类未实现造成失败。
- [ ] 新增 `gcnet_missing_m3/source_role_tokens.py`；`osram.py/model.py/train_gcnet.py` 透传 `osram_source_relation_layout`，默认 source-mean；role-tokens 是独立 opt-in，旧 state_dict/RNG 不变。
- [ ] 集成测试验证 disabled legacy、role-tokens zero-init 原 Flat 输出/RNG 相同、真实来源计算图连接；旧 mean 测试继续通过。
- [ ] 实验执行者仅扩展旧 worker/run/report 的 opt-in layout，复用原循环/保存机制。新目录 `experiments/osram_source_role_tokens_20261021/` 保存配置/运行记录；不可修改旧运行快照。
- [ ] 全部相关 CPU 测试、编译检查、原 Memory 核心 AST 对比；源码提交后部署不可变 Git 快照，验证 SHA256。
- [ ] biggpu 健康 GPU7 UUID 资源预检；MOSI/MOSEI 有限真实 batch smoke 通过后最多两个新训练并行；保留旧 MOSEI，不杀任何任务，不用坏 GPU4。
- [ ] 保存 smoke、PID、日志、源码 SHA、配置和启动记录，Git 提交/推送 github 当前分支。完整训练尚未结束时只报告 running，不称实验完成；持久队列完成后自动生成逐率/总体/高缺失报告。

## 文件责任

- 主代理：新模块、模块/集成测试、三处模型/config 路由、远程验证/启动、最终 Git。
- bounded executor：旧实验 worker/run/report 的 layout 扩展、新 runner 测试、新实验 README/状态报告；不碰主代理文件，不启动实验。

## 假设与限制

这一版是新的结构比较，不是已经证实的 bug 修复。取消早期平均同时改变 token 数、关系编码和新增参数量，不能把分数变化唯一归因于 mean；记录实测参数量与耗时。不预设 12-token 版有效，不将 Test-best 多 checkpoint 均值当单 checkpoint 泛化分数。
