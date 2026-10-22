# V3 三种连接方式：六组有限实验

本目录只安排 `head-preserving`、`source-slots`、`flat-hidden` 各自的 MOSI / MOSEI
seed66，共六组从头联合训练。三种连接方式互斥；原 Flat、standard V3 不重跑。

原协议保持：Flat1600、V3 width64 / pair128、residual 模式、CHM8×64、write_step0.6、
batch32、Adam、weight_decay1e-5、MSE、cyclic 缺失率0.0–0.7、`gcnet-random-mask-v1`。
MOSI 150轮，前100轮3e-4、后50轮3e-5；MOSEI 60轮，恒定1e-3。
按逐率 Test-best W-F1 严格改善选点，平分保留最早轮。八率均值是内部 Test-oracle
指标，可能来自八个不同轮次，不能称为单 checkpoint 部署表现。

## 执行

由主代理在 biggpu 的不可变源码快照内，用原 s0 Python 启动持久 controller：

部署只传代码时，还须提供原 `dataset/` 标签文件目录。已有导入链可能在设置
`GCNET_DATASET_ROOT` 前载入根 `config.py`，因此默认标签路径也必须指向同一份已核验
数据。本次使用已验证旧快照的 `dataset` 只读链接；MOSI/MOSEI pickle SHA256分别为
`931ee50f3a77b2eb18786840a880a439f3ae6542bf2dfb5e14ecc9690702af84`、
`0621d2cc93bee23642ef260f045f90c484136299d497df61a43527b34187f5d6`，
与原 manifest 指向的标签文件相同。原模态特征根目录仍由既有 manifest 校验。

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_v3_connection_screen_20261022.run \
  --source-commit COMMITTED_SHA --output /absolute/fresh/execution \
  --gpu-uuids HEALTHY_GPU_UUID_1 HEALTHY_GPU_UUID_2 --per-gpu 3
```

也可指定一张健康卡；`--per-gpu` 允许1–6，实际准入由显存决定。controller 先完成
全部六类真实数据 smoke，每卡同时最多一个 smoke；之后动态平衡正式任务并自动补位。
每次增并发前复查资源；要求可用 RAM≥64GiB、输出磁盘余量≥50GiB，显存保留
2GiB及实测 smoke 峰值的1.5倍，并为同卡在跑任务补齐尚未占用的峰值余量。
正式任务错开15秒启动。不更改 batch 或协议，不向其他任务发送信号。
物理 GPU4 及 `GPU-46fb379f-cc90-dc82-9b5e-5d011f552264` 均明确拒绝。

`QUEUE.lock` 防止同一队列重复控制；每个 worker 还有 `RUN.lock`。
`QUEUE.json` 保存任务身份、命令、PID/进程起始时刻、物理 GPU 与 UUID、状态和资源准入记录。
自动生成 `REPORT.json`、`PER_RATE.csv`、`RESULT.md`；失败项保留，未完成项不进入成绩表。
失败不会自动重试。所有 smoke 的结果仅证明协议和可运行性，不证明性能改善。

## 正确性门槛与产物

- 复用旧数据加载、训练、评测和持久化函数。只允许 V3 bool 和 variant 变化；旧配置
  仅补明确列出的历史默认字段。正式每轮与接受的 Flat66 对照核验 mask_trace、更新数和 LR。
- 复用 `osram_source_relation_attention_20261021.worker.REFERENCES` 内的 MOSI150 / MOSEI60
  Flat66，逐字节核对 test Mask、split、数据 hash、初始化。旧 MOSI bridge 只作历史校验，
  其权重不会作为训练初始值。
- 原模型参数和初始化 RNG 与新建 Flat 完全一致；每个缺失率首个真实测试 batch 的
  完整模型预测逐位一致。`flat-hidden` 依赖新建 Flat 的末层零初始化保证这一点，
  不声称加载训练过的 Flat 权重后仍等价，也不要求其新增 hidden 张量为零。
- 在第4次真实 optimizer update 后按模块 `expected_gradient_groups()` 检查累计有限非零
  梯度，并检查 SourceBase / SourceGap 输入路径；随后移除梯度 hooks。MOSI smoke为2轮×2批，
  MOSEI smoke为1轮×4批，评测各率最多1批。每训练 batch 只允许一次 CHM 扫描。
- 每轮保留八率分数、BEST、同次 prediction / labels / availability、完整 `last.pt`，
  包含模型、Adam、RNG、epoch、history、best ledger 和身份。smoke 实测完整恢复后
  optimizer / RNG / 输出一致。保存 source/data hashes、环境、GPU UUID、参数量、时间和峰值显存。

## 明确恢复

worker 中断后先核实 PID 和输出。兼容的完整训练状态可用原命令加 `--resume`；
`last.pt` 缺失、身份改变、Mask 改变或 BEST 产物损坏时拒绝恢复。完成项禁止重复运行。
恢复到 epoch 边界；未完成当前轮不被宣称保存。

controller 可用完全相同参数加 `--resume`：凭 PID、进程起始时刻及完整命令重新接管仍在跑的
本队列 worker；已退出的训练只有完整 `last.pt` 才重新排队，并显式传 worker `--resume`。
未通过或无完整产物的 smoke 不静默重跑。恢复不迁移服务器、不扩展任务。

## 本地测试

指定的 comp-repro 缺少 pytest；复用现有 multimodalerc310 的原 Torch2.2.2 环境并只借用
已安装 pytest，不安装依赖：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c \
  'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_v3_connection_screen.py"]))'
```

本地 CPU 测试覆盖有限任务、配置、学习率、坏卡拒绝、显存准入和报告边界；正式启动前
仍需原 s0 的六类真实 CUDA smoke。主代理负责部署、持久启动与最终验证记录。
