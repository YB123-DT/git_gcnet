# CAM统一消融目录

本目录统一管理原始Flat/CAM训练、严格checkpoint恢复和原版冻结Nested第二阶段
模型入口。`AUDIT.md`记录旧代码/环境陷阱，`PLAN.md`记录本轮范围。底层原模型与
sampler直接使用仓库原文件，不重新发明Memory数学定义。

## 本轮四组

| ID | 任务侧输入 | 目的 |
|---|---|---|
| A0 | Local | 不使用Context/Base或Missing-info/Gap |
| A1 | Local + Context | 检验Context增量 |
| A2 | Local + Context + Raw Missing-info | 保留Gap、关闭Query残差化 |
| A3 | Full CAM | 原始Residualized Missing-info |

四组都从头训练，不从旧BEST开始。相同初始化、数据、cyclic训练掩码协议、固定
测试mask、优化器和预算。保留相同4352维Flat输入及1600宽度，A0/A1将相应槽位
在输入LN之前置零。因此这是容量匹配消融，不能声称A0省去了Memory计算或参数。
Local Skip全部保留。未激活Gap保持硬零，联合写入和读写顺序不动。

执行范围：MOSEI seed66，60轮，constant LR1e-4，Adam，WD1e-5，batch32。
仅这四组；本轮不额外训练第二阶段GNN、不扩种子或数据集。

## 环境与数据

服务器固定biggpu，Python固定：

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python
```

Torch1.13.1+cu117、NumPy1.26.4、sklearn1.3.2。不能调用要求s0的旧worker setup。
数据沿用已接受MOSEI文件清单并校验SHA256；路径在导入trainer前绑定。
所有run保留完整生效TrainConfig、数据/源码/mask哈希、初始化摘要和环境。

原GCNet `gcnet-random-mask-v1`用于训练：host/guest分别采样，保留padded rectangle
规则和speaker选择，不改成conversation-mask-v2。测试复用现存合法mask并逐batch
核对labels/speakers/conversation IDs/utterance indices。

## 单组入口

在不可变源码快照根目录运行，真实UUID通过启动前nvidia-smi核对：

```bash
PY=/data2/yb/conda_envs/cmerc-repro/bin/python
$PY -m experiments.cam_unified_20261030.train \
  --variant A0 --gpu-uuid GPU_HEALTHY_UUID --source-commit SOURCE_SHA \
  --output /absolute/new/output/A0
```

将A0改为A1/A2/A3即可。`--smoke`只用于真实batch链路和资源测量，不产生成绩。
`--resume`只允许相同身份的完整LAST恢复，不把单独加载BEST权重叫作断点续训。

## 有限并行入口

先通过preflight，将PREFLIGHT.json写入独立输出根目录。快照包含SOURCE.json，
正式调度再次检查全部源码哈希。`bundle.py`只打包必要Python源码和本目录记录，
不复制数据、历史实验树、权重或feature cache。

```bash
$PY -m experiments.cam_unified_20261030.scheduler \
  --output /absolute/new/execution_v1 --gpus 7 --per-gpu 4 \
  --job-mib MEASURED_BUDGET_MIB --source-commit SOURCE_SHA
```

通过tmux持久运行。每卡最多4组，显存余量不足自动排队。物理GPU4和坏卡UUID
永远排除。失败不悄悄重跑，不杀其他人的进程。QUEUE.json保留PID/UUID/命令/退出码。

## 指标与选点

按用户指定逐rate Test-best W-F1，ACC/W-F1/MSE必须来自该次相同预测；并列取
最早epoch。训练MSE包含gold=0；二分类ACC/W-F1沿用原实现排除真实0标签。
保存每rate BEST、选点预测、精确mask身份，另保留一份完整LAST用于恢复，避免
每次epoch复制八份优化器状态。

**INTERNAL TEST-ORACLE SCREEN**：八率均值可能组合不同epoch权重，不代表
单checkpoint部署性能，也不作为独立测试泛化结论。共同mask只保证配对条件，
不会消除Test-oracle选择偏差。输出逐率结果、高缺失0.5/0.6/0.7均值及状态；
未完成或失败组不列为最终成绩。

## 第二阶段

`stage2.py`只复用原版159235参数Nested模型，不替换成容量搜索变体。
`infer.py`提供原Flat和原Frozen Nested的同mask推理入口。父Flat必须严格加载，
block-only权重必须匹配父checkpoint SHA256。原始LocalSkip不使用修正后的Local，
一次Memory扫描，冻结父模型但保留穿过原Norm/head的输入梯度。
第二阶段权重与第一阶段消融权重分别标注，不能混在一张从头训练消融表中。

具体启动/验证测量写入`RUN_RECORD.md`，结果由`summarize.py`生成JSON/CSV/RESULT。
代码、配置和小型记录可以提交；权重、数据、完整预测文件不上传Git。
