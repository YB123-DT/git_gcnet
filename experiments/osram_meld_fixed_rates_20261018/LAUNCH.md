# 八个固定率模型：实际启动记录

INTERNAL SINGLE-RATE TEST-ORACLE SCREEN — RUNNING，非最终分数。

已按用户“写完代码就启动”的授权部署并运行，没有等待二次确认。

- 服务器 biggpu；环境 `/data2/yb/conda_envs/cmerc-repro/bin/python`。
- 固定源码 commit：`2bdaf2cc5b0acc6eaf787274d59565c08aca0b02`。
- 不可变远程源码：
  `/home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/source_2bdaf2c`。
- 所有 worker 启动前验证相同 `SOURCE_MANIFEST.json` 中 135 个文件 hash；该文件
  副本在本目录保留。后续修改本地代码不影响运行中的快照。
- 输出：`/home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/runs_v1`。
- tmux：`meld_fixed8_seed66_1018`。
- controller 日志：上述实验根目录的 `controller_v1.log`。
- worker 日志：`runs_v1/queue/logs/R0.attempt1.log` 至 `R7.attempt1.log`。
- 数据：ECERC canonical pickle，不是刚跑过的 MMGCN 600维 Text。
- 已核对宿主 GPU0/5 的 UUID；GPU4 禁止，GPU6/7 原有任务未动。
- 启动时 GPU0/5 各约32GB空闲，输出分区约129GB空闲；每卡上限4组。

## 已发生的训练，不只是 Popen

2026-10-18 01:05:06 UTC 实际状态，完整证据见 `LAUNCH_STATUS.json`：

| Job | Train/Test rate | 物理 GPU | PID | 已提交轮数 | 状态 |
|---|---:|---:|---:|---:|---|
| R0 | .0/.0 | 0 | 2227949 | 5/50 | running |
| R1 | .1/.1 | 5 | 2228408 | 3/50 | running |
| R2 | .2/.2 | 5 | 2228411 | 2/50 | running |
| R3 | .3/.3 | 0 | 2228415 | 2/50 | running |
| R4 | .4/.4 | 5 | 2228422 | 2/50 | running |
| R5 | .5/.5 | 0 | 2228425 | 2/50 | running |
| R6 | .6/.6 | 5 | 2228431 | 2/50 | running |
| R7 | .7/.7 | 0 | 2228436 | 2/50 | running |

八组均已完成真实前向/反向/33次 optimizer step、同率选点和完整 epoch 提交；
所有模型参数量为原 Flat 的 13,290,795，初始化 hash 完全相同。队列 failure
列表均为空。每组 history 的 test_oracle 仅包含自己的 rate，EPOCH_STATUS
保留同率的 best 记录；选中预测为2610条同 epoch 原始测试样本，没有用
训练后新随机评估填 UA。

rate=.0 的实际第一轮同时完成准入检查，之后7组已自动放行，当前实际每卡4组。
没有新增独立 smoke 训练条件、扩充种子或启动其他数据集。

## 复现命令

在不可变源码目录内运行，需先部署对应 SOURCE_MANIFEST：

```bash
cd /home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/source_2bdaf2c
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_fixed_rates_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/runs_v1
```

实际通过 tmux 持久执行。输出存在时不能重复运行上面命令；恢复必须加 `--resume`，
并保持同一快照、配置、数据及完整状态。不要把旧启动状态文件当实时进度。

## 已验证与边界

- 本地：新 fixed 入口与 cyclic 回归、单率评测、exact-selection，共28项测试通过。
- 远程：八组均通过135文件校验、数据hash、GPU UUID绑定及原模型参数量检查。
- 首轮至当前轮：每组只有自己的 rate，33 batch/optimizer step、1038训练对话、
  JEPA loss=0，选点与完整恢复状态均落盘；八组初始化一致。
- 没有在本轮通过中途杀进程来重复验证恢复；复用现有经过验证的原子状态工具。
- 上述是运行/协议正确性证据，不能当作性能提升。最终还需50轮完成产物。
- 八个专用模型总预算高于一个 cyclic 模型，不能包装为等计算量改进。

后续队列自动汇总八率 ACC/UA/W-F1、八率均值与高缺失均值到 `RESULTS.json`，
包含失败记录；只在全部8组成功时标记 complete，然后停止，不追加搜索。
