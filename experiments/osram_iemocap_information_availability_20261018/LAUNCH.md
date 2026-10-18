# 实际启动记录

INTERNAL DIAGNOSTIC ONLY — RUNNING。不是最终结果。

用户授权：在 IEMOCAP 上做同样的 Local/Memory 检验；本轮不重训 OSRAM。
实验实现采用研究实验执行技能；冻结主干、沿用原数据/缺失/环境和精确测试
mask，而不是重新训练 baseline 或再添加模型模块。

## 固定版本

- 实验代码：`caa5bf07db8756793255c10511fed51d4a63f4fa`。
- 原主干数值代码：`785a01242ef1a82b97d5337acbb33ba963df5537`。
- 不可变实验快照：
  `/home/qinjinghui/remote_experiments/osram_iemocap_information_availability_20261018/source_caa5bf0`。
- 同一 SOURCE_MANIFEST.json 校验32个实验/helper文件。数值包另外绑定原训练
  checkout，检查原 Git SHA/无数值改动，并记录其逐文件 hash。
- 服务器 biggpu；环境 `/data2/yb/conda_envs/cmerc-repro/bin/python`。
- 原 IEMOCAP4/6 各5折 seed66，100轮新GCNet协议的80个 BEST 只读复用。
  所有80个选点 NPZ 均已远程核验存在且 SHA256 与原报告一致。
- 输出：
  `/home/qinjinghui/remote_experiments/osram_iemocap_information_availability_20261018/runs_v1`。

## 已发生的运行

2026-10-18 02:10:18 UTC 检查：

| 折任务 | GPU | PID | 实际进度 |
|---|---:|---:|---|
| IEMOCAP4 fold5 | 0 | 2384141 | rate0.0 冻结提取与精确选点回放通过，进入 Probe |
| IEMOCAP6 fold5 | 5 | 2384167 | rate0.0 冻结提取与精确选点回放通过，进入 Probe |
| IEMOCAP4 fold1 | 0 | 2384199 | rate0.0 冻结提取与精确选点回放通过，进入 Probe |
| IEMOCAP6 fold1 | 5 | 2384235 | rate0.0 冻结提取与精确选点回放通过，进入 Probe |

其余6个折任务在同一有限队列中，释放资源后自动补位。首轮检查4个worker
没有失败记录。此表是历史启动现场，不代表后来实时进度。

上述 EXTRACTED 日志只有在完整 train/test 提取、测试身份/预测/logits精确
核对、首句历史零、forward/backward维度、inactive Gap 和主干状态hash检查
全部通过后才写出。仍需 Probe 产物及完整 SUMMARY 才能声称任务完成。

宿主 GPU0 UUID：`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`；
GPU5 UUID：`GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`。每卡最多2个任务。
启动前 GPU0/5 各32GB空闲、输出分区127GB余量；本次实际进程显存约1.3–1.5GB。
GPU4 未使用，GPU6/7 原有任务未修改。

## 持久执行与产物

- tmux：`iemocap_probe_1018`。
- controller PID：2384061。
- controller 日志：实验根目录 `controller_v1.log`。
- worker 日志：`runs_v1/queue/logs/I{4,6}_fold{1..5}.attempt1.log`。
- 实时状态：`runs_v1/RUNNER.json`、`queue/QUEUE.json`、每折 `SUMMARY.json`。
- 所有折完成后自动生成 FINAL_SUMMARY.json / RESULT.md；失败不参与有效排名。
- Probe 权重/预测/特征保留远程，不上传 Git；只上传代码和小型证据/汇总。

实际执行命令（通过 tmux，SSH断开后仍持续）：

```bash
cd /home/qinjinghui/remote_experiments/osram_iemocap_information_availability_20261018/source_caa5bf0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_iemocap_information_availability_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_iemocap_information_availability_20261018/runs_v1
```

## 验证与边界

- 主线实测41项 CPU 测试通过；py_compile 和 git diff --check 通过。
- 原证据核查和新汇总/测试采用了有界 native 子agent；主线复跑集成检查。
- 4个真实GPU worker已完成首个rate的完整冻结提取与exact-mask预测核对。
- 此前 torch-sparse/libcusparse 的可选导入warning仍在，未更换环境；实际
  OSRAM forward 和精确回放已通过，不把warning当作训练成功或失败的依据。
- 恢复仅复用已完成的小头/率，不提供未完成Probe的逐step optimizer恢复。
  未完成rate缓存若被外部修改，现有恢复不会完整核验cache/probe全部来源；
  保留独立输出不作外部编辑，任何恢复前须重新核对缓存与selected checkpoint
  身份。不声称已验证进程中断恢复。本轮首跑不触发这一限制。
- 不自动追加主干种子、更多数据集、新模块或参数搜索。
