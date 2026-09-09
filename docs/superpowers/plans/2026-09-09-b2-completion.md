# B2 实现计划

> 使用 subagent-driven-development 执行独立 OSRAM 子任务，主代理负责集成与验收。

**目标：** Source-only pre-OSRAM completion 仅引导当前读；真实观测决定全部 persistent write。
**架构：** 冻结已有 .6 checkpoint 的 projectors/teacher，Stage1 只训练独立 predictor；Stage2 用 filled slots 的 source-status residual 生成 read_node，保留 observed_node 作 write_node。
**技术栈：** 现有 PyTorch / pytest / SSH biggpu 官方 Python，无新依赖。

## 固定边界

用户本轮只授权实现、tests、one-batch smoke。禁止完整正式训练与额外 control 实验。
保留 baseline、legacy completion、contextual predictor、write-step 与 retention 工具。
固定 causal OSRAM write_step=.6，沿用 checkpoint 的其他结构、loss、optimizer、cyclic mask。

## 任务

- [x] OSRAM：`gcnet_missing_m3/osram.py` 增加兼容 read_node/write_node 参数；key 用 write_node，query/local 用 read_node；不改 scan/update。`tests/test_osram_b2.py` 先验证缺少参数红灯，再覆盖每一步 memory 精确一致及 read 改变。
- [x] 组件：新 `gcnet_missing_m3/b2.py`，SourceOnlyM3Predictor 复用原 MMoE，逐 target 汇总真实 source mean；CompletedReadFusion 固定槽位+2D身份+zero-init residual。测试先验证缺失组件，再验证 mask/leakage/aggregation/gradient。
- [x] 模型：`model.py` 增加默认 none 的 completion_path；仅 B2 实例化新参数，保留旧模块；调用一次 source-only→read fusion→OSRAM；predictions_override 仅覆盖分类读，completion loss 始终用真实 predictor 输出。
- [x] Runner：独立 `gcnet_missing_m3/train_b2.py` 管理 Stage1 和 Stage2 checkpoint 转移，复用 train_gcnet 的数据/mask/loss/评价逻辑；`train_gcnet.py` 仅必要 config/transfer/预测记录 hook，不改 baseline 默认逻辑。
- [x] Stage1 audit：复用现有 `_metrics`，按 A/V/AV target 分组，两头分开，冻结目标及 projector 来源写 SHA256。固定 final Stage1 checkpoint，不用 audit 选点。
- [x] 验收：执行相关 OSRAM/JEPA/write/retention tests；真实 MOSI seed66 checkpoint one-batch Stage1 save/load→Stage2 两个 optimizer steps→test forward；不跑 epoch sweep。
- [x] 文档：记录参数数量、配置与阶段转移规则、每步 memory invariant、零初始化首步梯度例外、测试输出与 smoke。git diff --check，按 Lore commit 并 push 指定分支。

## 测试入口

在 remote repo 根目录执行：
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest tests/test_osram_b2.py tests/test_b2_completion.py tests/test_b2_training.py -q`

关键断言：`torch.equal(memory_reference[t], memory_changed_prediction[t])` 覆盖 normal/zero/random/shuffle，允许 read/logits 改变。打开融合末层或完成首步更新后检查 emotion→predictor 非零梯度；默认 zero-init 不应伪称第一步 predictor emotion gradient 非零。
