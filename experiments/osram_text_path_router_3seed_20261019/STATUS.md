# 三种子补跑：用户要求暂不启动

2026-10-19，用户批准“跑3seed看看”，随后加一组原MOSI Flat150配置上的版本。

本目录范围：MELD及IEMOCAP-6 fold5，seeds66/67/68，Flat和一阶段Router。
seed66复用已完成的四组结果，不重新训练；只新增seed67/68的八个独立作业。
原主干/Router数学、随机缺失、超参数、单一验证选点口径不变。
训练初始化、数据顺序及训练RNG随seed改变；划分和评价mask固定seed66。

实现源码：`218a0336cf41bcd741c84ae406269e1fc4ed9d7f`（已推送）。
已生成待传不可变archive，SHA256：
`c92f3c6d67665e80944bac8aeda1c7b0d9c48c90237e65b3cb5310a6a9187ecd`。

## 当前状态：PAUSED BEFORE LAUNCH

用户后续明确要求“先别启动吧是内存不够了吗”。SSH已恢复；当前暂停是用户
指令，不再是连接故障。没有训练启动命令、新GPU占用或新队列；不自动恢复。
核查远程 `/proc` 后未发现本轮新增训练进程，只有上述source archive已上传。
资源只读检查见 `RESOURCE_CHECK.json`。

## 此前连接故障记录

新增训练 **NOT STARTED**，没有新GPU占用、PID或训练成绩。
biggpu最后一次资源检查可见健康GPU1、3空闲，GPU4被排除；这不是资源预留，
恢复连接后必须重新检查。随后旧SSH复用通道与多次独立`-S none`连接都无响应，
独立连接报`Connection timed out during banner exchange`。
TCP22可连接但5秒内没有SSH banner，不能据此声称SSH登录可用。
没有跨服务器转跑、重启服务器、杀其他进程或静默改协议。

源码与本地测试已完成，尚缺远程数值/运行检查和正式启动。不可将本状态写为running。

## 仅在用户重新要求启动后的有限命令

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
