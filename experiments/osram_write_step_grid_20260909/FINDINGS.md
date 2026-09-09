# Frozen write-step grid：结果与阶段决策

INTERNAL TEST-ORACLE CHECKPOINT DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

## 结论

两个数据集的五率平均 W-F1 都在当前 grid 内继续改善到 eta=0.8。
**没有观察到 0.9 附近的内点最优，也没有找到最优步长。**
按运行前锁定的规则，本轮不启动 eta=0.9 重训练，不替换成其他训练点，
不追加小于 0.8 的参数，不加 adaptive gate 或其他模块。

## 五 seeds × 五 rates

Seeds 66–70；missing rates 为 0.0/0.1/0.3/0.5/0.7。
先对每个 seed 的五率等权平均，再统计五 seeds 的均值和样本 SD。
下表为 W-F1（%），**不是八率均值**。

| 数据集 | eta=1.0 | eta=0.95 | eta=0.90 | eta=0.80 |
|---|---:|---:|---:|---:|
| IEMOCAP-4 | 81.6240 ± 0.5384 | 81.7941 ± 0.5343 | 81.8668 ± 0.5197 | 82.1638 ± 0.5136 |
| CMU-MOSI | 80.0664 ± 0.5412 | 80.1368 ± 0.5061 | 80.1445 ± 0.3653 | 80.2006 ± 0.3411 |

- IEMOCAP-4：三个相邻步长变化都在 5/5 seeds 为正；eta.8 相对1.0
  提升 +0.5398 个百分点，eta.8 相对.9 提升 +0.2970 个百分点。
- MOSI：eta1→.95 在 3/5 seeds 为正，.95→.9 在 2/5 为正，.9→.8
  在 5/5 为正。eta.8 相对1.0 平均 +0.1342 个百分点，仅 3/5 seeds
  为正；不是每个 seed 都单调，也不是所有 rate 都改善。
- MOSI 的 miss=.1、.5 均值在 eta.8 低于 Reference，不能说高鲁棒性
  问题已被普遍解决。完整 per-rate/per-seed 表见 `SUMMARY.md` 和 CSV。

这说明弱化当前写入值得关注，但“在已测区间继续改善”不等于
“写入严重过强已被证明”，也不等于“eta.8 是最佳超参数”。目前观察到的
仍是用 eta1 训练的 checkpoint 在推理时受到扰动后的结果。

## E_old / E_new：不能把拟合误差等同于任务收益

E_old = 历史 missing-slot probe 的 `err_decay`，即当前实际读取前的
memory association error；排除 NO_HISTORY。E_new = 当前 observed
slot 的 `err_after`，即施加这次 write 后的相对拟合误差。

以下为 miss=.7：run 内按有效 slot/head 记录平均，再对五 seeds 等权。

| 数据集 | eta | E_old | E_new |
|---|---:|---:|---:|
| IEMOCAP-4 | 1.00 | 0.207436 | 0.001865 |
| IEMOCAP-4 | 0.95 | 0.223855 | 0.044981 |
| IEMOCAP-4 | 0.90 | 0.244474 | 0.086785 |
| IEMOCAP-4 | 0.80 | 0.290742 | 0.167171 |
| MOSI | 1.00 | 0.132166 | 0.001097 |
| MOSI | 0.95 | 0.144266 | 0.026211 |
| MOSI | 0.90 | 0.159051 | 0.050920 |
| MOSI | 0.80 | 0.192462 | 0.099875 |

两种误差在该 rate 都随着写入减弱而增大，但 F1 均值仍改善。
因此不能把这次收益解释为“降低了历史 association error”，也不能声称
已经找到 E_old 与 E_new 的最佳折中点。一个可能的解释是弱化写入改变了
表示的平滑/尺度/近时观测追踪程度，但本轮没有区分这些机制，更没有证明
哪一种解释导致 F1 提升。

当前 write 的 post 状态只能影响未来 utterance；不把 E_new 直接归因为
当前样本的分类误差。完整按 rate、模态分组的 E_old/E_new 已保存，不
将 miss=0 未定义的 E_old 补零参与平均。

## 重训练决策

预先规则要求两个数据集都满足 eta.9>1（均值为正且至少3/5 seeds正），
并且 eta.8 均值低于 eta.9，才启动固定 eta.9 的重训练。

前一条通过，但后一条两个数据集都未通过：

| 数据集 | .90−1.0（百分点） | 正 seed 数 | .80−.90（百分点） | Gate |
|---|---:|---:|---:|---|
| IEMOCAP-4 | +0.2428 | 5/5 | +0.2970 | 不通过 |
| MOSI | +0.0782 | 3/5 | +0.0560 | 不通过 |

`gate.json` 保存该判定。没有新训练进程，也没有新增 trainable eta、
optimizer 配置或 production-backbone 变更。这不是“重训练失败”，
而是条件未满足，重训练尚未执行。

## 复现与验证

- IEMOCAP4 五个已有 checkpoint epochs：33/54/38/30/64。
- MOSI 五个已有 checkpoint epochs：54/49/30/61/50。
- 每个 seed 一个固定 `best.pt`，沿用历史八率均值 Test-oracle 选点；
  没有用不同 rate 的独立最佳 epoch 表替代。
- 200 个完整 grid cells：50 个 IEMOCAP4 eta1/.9 从上一轮直接继承；
  150 个新增 evaluation-only cells；没有重跑这些继承点。
- 跨继承/新结果的 checkpoint SHA、config、epoch、selection metadata、
  mask hash 匹配；新评估检查全部 state-dict 参数/缓冲量未变。
- 203 个相关模型/诊断测试通过，12 个新旧纯统计测试通过。
- 450 份新增 gzip JSONL，共 4,128,104 条记录已校验；记录数量与
  metadata 一致，fit_gain 恒等式、observed-only mask、有限值检查通过。
- `git diff --check`、语法编译通过。没有新增依赖。

代码改动仅在现有 inference helper/runner 中开放两个锁定 fixed 模式
以及 `--modes` 子集，以继承已有结果；新汇总脚本生成曲线和错误指标，
不修改 `osram.py`、主模型、训练器、JEPA 或 loss。

GitHub 保存代码、每组 summary/metadata、CSV、协议和 SHA256 manifest。
约 259 MiB 原始 gzip 保留在本地当前目录及 biggpu：
`/data2/yb/remote_experiments/osram_write_step_grid_20260909/`。
`RAW_MANIFEST.csv` 列出 450 个新增文件；继承原始记录的来源另见
`provenance.csv`，不重复复制或删除。全部只用于内部诊断。
