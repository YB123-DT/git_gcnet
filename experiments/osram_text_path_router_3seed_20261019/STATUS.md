# 三种子补跑：全部完成并汇总

## 最新结果

八个新增作业全部完成，seed66四个既有作业未重训。三种子、逐率和高缺失
结果已保存于RESULT.md、METRICS.csv与精简SUMMARY.json。
来自12个run的96个评价NPZ已重新计算ACC、UA与W-F1，与记录误差小于1e-9 pp。
原始源码、初始化和配对训练/验证/测试mask检查全部通过。
本轮为单一validation checkpoint的八率评价，不是历史逐率Test-best；
IEMOCAP只有6类fold5，不是五折平均。MOSI不在本完成范围内。

## 最新实测：2026-10-19 14:30–14:34 UTC

SSH已恢复，远程RAM准入队列成功运行。可用RAM从此前约1.1GiB恢复至
337–340GiB；Swap仍接近满，不代表当前RAM依然不足。MELD四组在物理GPU3、
IEMOCAP-6 fold5四组在物理GPU1，均已出现真实epoch日志，未重复运行seed66。
数值源码为`00b75520eaa5f3623379a72b7a619d2aa85a96ce`。

| 数据集 | seed67 Flat / Joint PID | seed68 Flat / Joint PID | 已观察进度 |
|---|---|---|---|
| MELD | 2735675 / 2735690 | 2735697 / 2735704 | 1–2/50 |
| IEMOCAPSix fold5 | 2735708 / 2735716 | 2735720 / 2735727 | 3/100 |

远程tmux为`router_ram_00b7552`，分类队列PID2735119。状态与逐作业日志位于：
`/home/qinjinghui/remote_experiments/osram_text_path_router_3seed_20261019/execution_v1/`。
MOSI独立队列已执行preflight但失败，尚未训练，详见对应STATUS；不能将其计入
已启动训练数量。下文未启动/等待描述是历史记录，不代表本次最新状态。

### 数据加载与RAM核查

`dataloader_meld.py`通过`pickle.load`读入整个特征文件，并保存全划分的话语
特征数组；IEMOCAP/MOSI的Dataset同样将预提取特征组装在CPU内存中。
不是加载原始视频。MELD host/guest引用同一数组，不是完整复制两份。
每个独立训练进程各自加载Dataset；三个split loader引用同一Dataset。
本轮分类`runner_helpers.py:setup`及MOSI runner均指定`num_workers=0`，
不会额外创建`pt_data_worker`。仅当前batch搬到GPU。

本轮已运行分类进程的RSS约2.4–2.6GiB。另一个现存PID2483243所属账号
`qinjinghui`，入口为`/data2/cby/Incomplete_fake_detection/train_fanvm.py`，
RSS约116GiB、匿名RSS约116GiB；不是本轮Router实验。账号不等于实际操作者身份。
此前瞬时118GiB级`pt_data_worker`已不在本次top进程列表，未确认其父进程，
不能据此归属于该脚本，也不能把多个RSS相加或声称发现内存泄漏。
没有停止或修改任何其他任务。

2026-10-19，用户批准“跑3seed看看”，随后加一组原MOSI Flat150配置上的版本。

本目录范围：MELD及IEMOCAP-6 fold5，seeds66/67/68，Flat和一阶段Router。
seed66复用已完成的四组结果，不重新训练；只新增seed67/68的八个独立作业。
原主干/Router数学、随机缺失、超参数、单一验证选点口径不变。
训练初始化、数据顺序及训练RNG随seed改变；划分和评价mask固定seed66。

实现源码：`218a0336cf41bcd741c84ae406269e1fc4ed9d7f`（已推送）。
已生成待传不可变archive，SHA256：
`c92f3c6d67665e80944bac8aeda1c7b0d9c48c90237e65b3cb5310a6a9187ecd`。

## 当前状态：授权恢复；模型未启动，等待资源

用户此前要求暂停，随后明确两次要求“启动”，已撤销暂停。2026-10-19
14:03UTC复查：RAM502GiB中约500GiB已使用、MemAvailable约1.6GiB，Swap
几乎全满；GPU1/3/7空闲。与13:33的285GiB可用RAM不同，不能沿用旧准入结论。
不在此时硬启模型。标准库轻量资源队列要求MemAvailable至少64GiB、连续两次
检查满足后才解压源码并启动已授权队列；两组队列之间重新检查。64GiB是
保守工程余量，不是模型实测CPU峰值或算法超参数。不清理Swap/cache，不杀进程。
后台资源队列状态、实际PID和当前等待/训练状态以随后启动记录为准，不把等待
写成模型训练。先前资源证据仍保留在 `RESOURCE_CHECK.json`，不覆盖成新状态。

本地持久化部署恢复进程现已启动：tmux `osram_deploy_00b7552`，Python PID4122666。
2026-10-19 14:13UTC核实存活，状态waiting_for_ssh；远程队列尚未确认部署，模型
尚未训练。最多60次连接/部署恢复尝试，间隔60秒；一旦部署成功，远程RAM准入
队列接手。完整路径、源码及资源证据见 `RESUME_LAUNCH.json`。不得删除其引用的
本地source archive和脚本快照，否则后台部署将失败。

## 此前连接故障记录

新增训练 **NOT STARTED**，没有新GPU占用、PID或训练成绩。
biggpu最后一次资源检查可见健康GPU1、3空闲，GPU4被排除；这不是资源预留，
恢复连接后必须重新检查。随后旧SSH复用通道与多次独立`-S none`连接都无响应，
独立连接报`Connection timed out during banner exchange`。
TCP22可连接但5秒内没有SSH banner，不能据此声称SSH登录可用。
没有跨服务器转跑、重启服务器、杀其他进程或静默改协议。

源码与本地测试已完成，尚缺远程数值/运行检查和正式启动。不可将本状态写为running。

## 已重新批准的有限命令

将上述不可变source传到独立remote source目录，校验archive与seed66模型源hash。
在cmerc-repro环境使用：

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -m \
  experiments.osram_text_path_router_3seed_20261019.run \
  --output NEW_REMOTE_OUTPUT --source-commit 218a0336cf41bcd741c84ae406269e1fc4ed9d7f \
  --gpus GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
```

GPU顺序为MELD/IEMOCAP，每卡最多本任务4组。队列会按已测峰值与余量准入；
资源不足应换健康卡或等候，不改batch/算法来硬塞。结束后自动合并seed66并汇总。
MOSI由独立`osram_text_path_router_joint_mosi_20261019`队列处理，不在这里扩展种子。
