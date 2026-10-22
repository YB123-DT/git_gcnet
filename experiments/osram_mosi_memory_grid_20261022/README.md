# MOSI Flat：写入强度 / 单 Head 维度双组调参

INTERNAL TEST-ORACLE SCREEN

只用原 cfg84 no-JEPA LargeFlat；不修改模型公式，不加新模块或新 Loss。
本任务替代前面尚未确定设计的 Memory 改版讨论，不启动写入保护模块。

| 调参组 | 五个配置 | 其他 Memory 设置 |
|---|---|---|
| 写入强度 eta | 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 8 heads，K=V=64 |
| 每个 head 的大小 | K=V=32 / 48 / 64 / 96 / 128 | 8 heads，eta=0.6 |

两组共享同一份本轮从头训练的eta0.6/K=V64对照，共9个独立训练，
不是对9个配置做联合网格搜索，也不自动追加种子或选出赢家后继续调参。

## 固定协议

- MOSI，官方原train/dev/test划分，seed66。
- 150epochs，前100轮LR3e-4、后50轮3e-5；batch32、Adam、WD1e-5。
- s0环境：`/data2/yb/reproduction_workspace/envs/s0/bin/python`，
  Torch2.2.2+cu121、NumPy1.24.3、sklearn1.3.2。
- 原`gcnet-random-mask-v1`，host/guest、batch padded rectangle、cyclic0.0–0.7不改。
- 原MSE、Local/LocalSkip、联合BlockDeltaWrite、read-before-write、Query及Flat读出。
- 原LargeFlat有效隐藏和输出均1600；不启用V3、Nested、Gate、JEPA或其他模块。
- 八率分别按Test W-F1严格提升选BEST，平分保留最早轮；ACC/W-F1/MSE取同次预测。
  这是内部Test-oracle探索，不能当单checkpoint部署性能或独立泛化成绩。
- 测试Mask与已接受原Flat66的exact test bank严格一致；训练每轮核对原Mask轨迹、
  更新次数和LR，不因模型维度变化而换Missing协议。

## Head维度改变的解释边界

Key/Value维度同时变化，而Head个数固定8。有效前向读出宽度为8*d，
context接口为2*8*d（因果backward半边仍为零），因此Flat输入宽度
`256+4*(2*8*d)`也改变；不额外插入压缩/补齐层。Flat隐藏1600、输出1600不变。
所以这不是参数量匹配的纯存储容量因果实验：K/V/Query投影和Flat第一层输入
参数量也会随之变化。报告全部参数量、Flat参数量、状态元素数与输入宽度。

所有模型从头训练。为避免改构造器形状导致无关层随机初始化漂移，形状相同的
原有参数/缓冲区从**同seed未训练原Flat**初始化复制；形状变化参数保持各自
fresh seeded初始化。不是加载训练好的checkpoint，不是两阶段，也不是续训。
写入强度组全部参数形状相同，初始权重hash必须与已接受原Flat初始化完全一致。

## 执行与资源

复用原数据hash与旧模型预测桥接检查。每配置做两轮、每轮最多3个真实训练batch的
最小smoke，核对梯度有限、真实参数更新、单次Memory扫描、完整Adam/RNG/预测恢复。
smoke不作为实验成绩；正式训练独立从零初始化，不接着smoke权重训练。

biggpu健康GPU7/GPU0，各最多5个本任务训练进程。队列先实测显存峰值，再按原
准入规则预留每个进程峰值*1.5及2GB设备余量、64GiB系统可用RAM、50GiB磁盘空间。
容量不足自动等待，不改batch或超参，不杀其他任务。物理GPU4及坏UUID禁止。

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_mosi_memory_grid_20261022.run \
 --output /data2/yb/remote_experiments/osram_mosi_memory_grid_20261022/execution_v1 \
 --source-commit SOURCE_SHA \
 --gpu-uuids GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
 --per-gpu 5
```

通过tmux持久运行；每个模式独立日志/输出，队列写`QUEUE.json`。
每轮保存八率BEST、同次预测/Mask、参数/源码/环境/数据hash和完整`last.pt`恢复状态；
checkpoint全部保留，不提交权重。`worker --resume`只接受身份完全匹配的完整恢复；
控制器不会静默重复失败作业、换参数重试或自动重启已有队列。
`report --output OUTPUT`只做离线汇总，不新增推理；只有完成150轮并有八率结果的
作业进入最终表，两张表的差值均基于本轮共享baseline。进行中、失败、未启动单列。
