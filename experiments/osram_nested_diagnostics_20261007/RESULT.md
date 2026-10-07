# Nested 四数据集诊断：已启动

INTERNAL DIAGNOSTIC ONLY。当前为运行记录，不是最终结果表。

模型固定为旧版 `nested_gnn_rooted_evidence`；无训练、无 Flat 对照，现有 Test-oracle BEST不变。MOSI/MOSEI 3seeds；IEMOCAP4/6 3seeds×5folds；合计36源run、540个 dataset/seed/fold/condition评测条件。每condition可含多个batch，不能把540称作实际Memory scan总次数。

## 启动证据

- 所属服务器：biggpu (`user22`)；物理GPU6 / `GPU-e4cafb17-818e-216a-b94a-7440063a9153`。白名单不含坏GPU4。
- 活跃根目录：`/data2/yb/remote_experiments/osram_nested_diagnostics_20261007/attempt2`。
- 运行代码：`557c92537357400a321b09b8340caa7d07807feb`，独立immutable source。
- 持久tmux：`osram_nested_diag_attempt2_20261007`；dispatcher PID2385841。
- 首批MOSI seed66 PID2385858、IEMOCAPFour seed66 fold1 PID2385867，GPU6最多2组。其余34源run按资源自动补位。
- `SOURCES.json`：36源config/metrics、288checkpoint、288原预测文件SHA256均核对。
- `LAUNCH.json`：启动时队列快照；远程 `DISPATCH.json` 才是实时状态，不把该静态快照当最终完成证据。
- 12项远程CPU评测测试通过；包括真实Nested非零decoder屏蔽、首句和padding、冻结配置device复制、有限队列和未完成汇总。

## 已经通过的真实重放检查

MOSI seed66的随机八率 Full已经复现原选定Nested预测与指标：

| Rate | Full W-F1 (%) |
|---|---:|
| .0 |88.077881|
| .1 |86.357894|
| .2 |83.735235|
| .3 |80.522617|
| .4 |81.011858|
| .5 |77.675491|
| .6 |75.031657|
| .7 |75.524540|

这是原checkpoint重放检查，不是新训练成绩或新增收益。IEMOCAPFour seed66 fold1已进入真实评测并至少完成随机.0–.5；每condition都强制检查原分类、mask/label一致和state不变。固定组合及全部源run尚未完成，暂无四数据集最终均值。

## 实验定义与失败记录

本轮按六项诊断运行；Base/Gap干预同时移除Nested输入内容/节点和最终对应槽，Local-off同时清理Nested Local输入、最终Local槽与整个Skip输出。干预作用范围不同于旧Flat输出槽开关；结果不直接冒充同一模型的旧诊断值。

Query/Addressing/Read余弦针对原scan中Nested之前的张量；raw query和residualized query区分记录。首句零读值余弦未定义，不填0。

首次代码1910381的队列在配置设备赋值阶段失败，没有模型推理或性能分数。失败日志保留，父进程停止，相关进程已退出；修正冻结配置问题后启动独立attempt2，未覆盖失败产物或改训练权重。后续调度遇失败停止补位，不对同一原因自动重试。

完成后由 `summarize.py` 自动汇总各rate、overall/high、七固定组合、读出开关的W-F1/ACC以及IEMOCAP UA。源预测、权重不覆盖；不把尚缺fold/seed的结果称三种子完整均值。Query per-head/target CSV和undefined counts保留在各run。
