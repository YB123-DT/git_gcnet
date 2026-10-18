# 实际启动：MELD Flat 小学习率 / 大 weight decay

INTERNAL TEST-ORACLE SCREEN — RUNNING，不是完成结果。

用户澄清本次调参对象为 MELD 原 Flat，而非 IEMOCAP Probe。八组均从头训练，
只改 LR/Adam weight_decay，原数据、随机缺失、环境、结构与50轮constant预算不变。

## 运行版本及位置

- 实验代码：`a879ff8d66b8be6abf5804b34b1ca44d6af199a5`。
- 原数值代码：`b4a313841f098487a6482d2712bdac94aa8e319e`，未修改。
- 不可变快照：
  `/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/source_a879ff8`。
- SOURCE_MANIFEST.json 记录30个实验/helper/config文件hash，worker启动时校验。
- 输出：
  `/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1`。
- biggpu，`/data2/yb/conda_envs/cmerc-repro/bin/python`。
- tmux：`meld_optimizer_1018`；controller PID2441448。
- controller 日志：实验根目录 `controller_v1.log`。
- worker 日志：`runs_v1/queue/logs/O{0..7}.attempt1.log`。

## 已启动八组

| Job | LR | weight_decay | 物理GPU | PID |
|---|---:|---:|---:|---:|
| O0 | 2e-4 | 2e-4 | 1 | 2441562 |
| O1 | 2e-4 | 1e-3 | 2 | 2441566 |
| O2 | 1e-4 | 2e-4 | 1 | 2441579 |
| O3 | 1e-4 | 1e-3 | 2 | 2441586 |
| O4 | 3e-5 | 2e-4 | 1 | 2441642 |
| O5 | 3e-5 | 1e-3 | 2 | 2441668 |
| O6 | 1e-5 | 2e-4 | 1 | 2441684 |
| O7 | 1e-5 | 1e-3 | 2 | 2441701 |

启动前核对GPU1/2各约32GB空闲、输出分区126GB余量。通过真实UUID绑定，每卡
4个独立进程；早期nvidia-smi显存约1300MiB/进程（含CUDA上下文/缓存）。
GPU4禁止，GPU0/5现有IEMOCAP Probe和GPU6/7其他任务均未修改或停止。

## 检查及结果口径

首轮已核实：八组均已完成真实训练、反向、Optimizer step及完整epoch提交，
不是只有PID；实际LR/WD与各组配置相符，初始化与W0相同，所有已提交轮的
train和八率Test mask stream与W0一致，无失败。完整首次现场见
LAUNCH_STATUS.json，其中O6的0.0/0.7选点权重和对应预测还做了hash/epoch/
mask/ACC/UA/W-F1复核。该文件不是实时进度或最终分数。

- 62项CPU配置/实际Optimizer/实际mask审计及已有训练capture回归测试通过。
- compile及whitespace检查通过；旧训练器没有组name，审计按真实参数归属
  标记副本，不修改真实Optimizer的结构、顺序或参数。
- 每组首轮及后续轮实际LR/WD、模型初始化hash和train/八率test mask stream
  由运行观察器核对，不把相同seed当作配对证据。
- 准入复用已完成原Flat真实显存记录，无额外smoke训练条件。真正的完整
  epoch提交、checkpoint和选点产物由各组EPOCH_STATUS.json证明。
- 同源码、同环境、同seed的已完成W0只读复用，八率ACC60.838123 /
  UA35.632827 / W-F1 56.788705；不重新跑已完成对照、不替换历史三种子基线。
- 保留每率ACC-best权重及同次forward的predictions/mask/UA/W-F1，不使用
  最后一次新随机评估补UA；完整last包括optimizer、RNG和实际LR记录。
- 全部50轮完成后自动生成RESULTS.json / RESULT.md，然后停止，不追加搜索。

## 复现

```bash
cd /home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/source_a879ff8
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_optimizer_screen_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1
```

已经启动的输出不要再次运行；恢复必须显式--resume并保持来源/配置/数据不变。
实时进度查RUNNER.json、queue/QUEUE.json和逐组EPOCH_STATUS.json，本文件保留
启动现场，不冒充当前进度或最终分数。
