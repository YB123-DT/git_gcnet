# B2 执行入口（当前只执行 smoke）

工作目录：`/data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple`，服务器 `biggpu`。
Python：`/data2/yb/reproduction_envs/s0/bin/python3.10`。
本轮没有启动 P0/P1 正式训练。P0 直接继承当前 causal .6 checkpoint，不重跑。

## 已执行 one-batch smoke

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
 /data2/yb/reproduction_envs/s0/bin/python3.10 -m gcnet_missing_m3.train_b2 \
 --stage smoke \
 --base-checkpoint /data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_66/best.pt \
 --feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features \
 --output-dir /data2/yb/remote_experiments/osram_b2_20260909/smoke_seed66 \
 --device cpu
```

输出目录非空会拒绝覆盖。生成的 `.pt` 标识 epoch0/smoke，不能拿来当完整 Stage1 结果。

## 待用户确认后才执行的训练

两个 stage 都用同一个 base checkpoint 和 seed。Stage1 的 `--stage stage1` 入口只运行 frozen-space predictor 训练，训练 budget 默认继承 base，可显式传 `--epochs`。完成后生成 `source_only_completion_pretrain.pt` 和独立 latent audit。

Stage2 的 `--stage stage2` 入口需要再指定 `--pretrain-checkpoint`，从 base 继承全部训练/评价配置，仅改变 B2 path 与必要的初始化来源；允许显式 device/epochs。禁止默默改 LR/loss/rate/selection policy。

Stage2 直接复用 `train_gcnet.run_experiment`，train mask 保持 cyclic，classifier/JEPA/EMA/selection/最终各 rate prediction 保存均沿用基线。当前 base 使用一个八率均值 Test-oracle checkpoint；本轮没有擅自变成逐 rate 选 epoch。

未来启动前需确认 seed 范围、预算以及是否沿用该内部诊断 selection；本次不自动扩到五种子。

## 验证入口

在本地有 Git 历史的 worktree 提取下列两个历史文件，以环境变量传到 remote pytest：

```bash
OSRAM_HISTORICAL_SOURCE_B64=$(git show d27b58f:gcnet_missing_m3/osram.py | base64 -w0)
OSRAM_B2_HISTORICAL_SOURCE_B64=$(git show fca2aa3:gcnet_missing_m3/osram.py | base64 -w0)
```

remote 同步目录没有 `.git`，必须提供这两个变量；否则旧历史回归测试会失败在 git show，而不是模型行为。

已验证 suites：test_osram、test_osram_b2、test_osram_write_step、test_missing_m3、test_memory_retention、test_memory_retention_analysis、test_write_intervention、test_write_intervention_summary、test_write_step_training_plan、test_write_step_grid_summary、test_write_step_range_summary、test_write_step_cross_summary、test_write_step_evaluation、test_b2_completion、test_b2_training。

初次红灯来自缺失 B2 API；转移配置的红灯捕获 top-k mismatch 未拒绝，已修复。最终结果见验证记录。全程无新依赖；CPU FP32 是本轮验证设备，未宣称通过 CUDA/AMP。
