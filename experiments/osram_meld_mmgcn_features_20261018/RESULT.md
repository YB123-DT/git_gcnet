# MMGCN MELD 特征版原 Flat：完成结果

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC

已在 biggpu 完成 seed66、50/50 epoch，SUMMARY=complete，训练 PID 已退出且无错误。
本次核验没有重训或新增随机推理，只读取训练时保存的选点预测。

## 八率主结果

每率按测试 ACC 选最佳 epoch；UA/W-F1 必须来自该 epoch 同一份预测与 mask。
这不是一个单独 checkpoint 的八率部署成绩。

| Missing rate | Selected epoch | ACC (%) | 对应 UA (%) | W-F1 (%) |
|---|---:|---:|---:|---:|
| .0 | 17 | 61.264 | 33.256 | 57.072 |
| .1 | 17 | 60.575 | 32.423 | 56.237 |
| .2 | 20 | 58.927 | 32.273 | 55.126 |
| .3 | 40 | 58.391 | 30.485 | 53.894 |
| .4 | 37 | 57.433 | 31.979 | 54.370 |
| .5 | 34 | 56.169 | 29.762 | 52.248 |
| .6 | 36 | 55.134 | 27.977 | 50.270 |
| .7 | 28 | 54.713 | 26.343 | 49.101 |
| 八率均值 | — | **57.826** | **30.562** | **53.540** |
| 高缺失 .5/.6/.7 | — | **55.338** | **28.028** | **50.540** |

八率平均 CE=1.301448，高缺失 CE=1.351140。

## 与既有 ECERC 参考比较

采用用户接受的原 Flat **seed66** 参考，而不是拿单种子结果与三种子均值混比。
参考出处：`experiments/osram_meld_testbest_ua_20261014/BASELINE.md`。

| 特征来源 | ACC | 对应 UA | W-F1 |
|---|---:|---:|---:|
| ECERC 原 Flat，seed66 | 60.675 | 35.436 | 56.807 |
| MMGCN 原 Flat，seed66 | 57.826 | 30.562 | 53.540 |
| MMGCN − ECERC (pp) | **−2.849** | **−4.874** | **−3.267** |

该比较是历史参考，不是逐参数初始化/逐样本 exact mask 完全相同的严格配对。
Text 输入维度改变会改变部分参数及初始化 RNG 消耗；双方 best epoch/mask 也可能不同。
ECERC 的历史数值源码为 b4a3138；本次沿用此前特征切换试验的不可变 737fdb1
数值源码和 worker，来源均明确记录，不把它们称作同一源码新跑的成对对照。
两者保留相同 Flat/Memory 结构、原训练超参、环境及 GCNet random missing 协议。

**本固定配置与 seed 下，更换为该 MMGCN 特征没有改善 ACC、UA 或 W-F1。**
不据此推断所有配置下的普遍优劣，也没有追加调参或三种子训练。

## 实际改变了什么

下载固定的 MMGCN 作者仓库 release，逐句比对全部 13,708 句话：

- Audio：300维，原始及 float32 数组与 ECERC 完全相同。
- Visual：342维，原始及 float32 数组与 ECERC 完全相同。
- Text：MMGCN 600维，ECERC 1024维，是本次实际不同的特征。
- 当前 train/dev/test、标签、speaker、句子及顺序均保留；没有照搬上游合并 train/dev 池。
- 视觉全零位置仍为 train3407/dev95/test258，不能声称本轮解决了全零视觉问题。

详见 `PREPARED.json`。本次唯一预期的数据变化是发布的 Text 表征；没有改变
Missing、Memory 联合写入、读写顺序、Query、Flat1600、Local Skip、分类头或 loss。

全部八率 ACC-selected checkpoint 的 Fear/Disgust Recall 均为 0，而测试集
实际分别有 50/68 条这两类话语。这是当前选点结果中稀有类别识别弱的描述，
不自动定位为特征损坏、Memory 写入失败或某个模块错误；完整七类 recall/support
已保留在 `FINAL_SUMMARY.json`。

## 配置、运行与保留

- 固定启动实现：`9c9ec92ae82927e5970258e81b794dc2dc959377`；配置见 `CONFIG.json`。
- 原环境：`/data2/yb/conda_envs/cmerc-repro/bin/python`，Torch1.13.1+cu117、NumPy1.26.4。
- 原 seed66、50 epoch、batch32、Adam constant LR=.001、weight decay1e-5、dropout=.5。
- 原 `gcnet-random-mask-v1`、cyclic0.0–0.7、emotion-only/no-JEPA，未加入 cosine 或 Block2。
- 健康物理 GPU5，UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`；GPU4 未使用。
- 参数量 13,181,403，比 ECERC 少 109,392，仅对应实际 Text 输入维度变化。
- 正式 worker 总耗时 1240.230 秒，约 20分40秒；峰值 CUDA allocated
  582,855,168 bytes。不是整卡显存占用，也不是独占机器方法速度基准。
- 三 batch smoke 是独立进程，7.438秒；不纳入性能结果，正式训练重新初始化。
- 远程输出：
  `/home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018/runs/Flat_MMGCN_seed66`。
- 保留八个 BEST、原始与复制的 exact selected 预测/mask，以及完整 `last_training.pt`。
  恢复状态含 model/optimizer/scheduler/scaler/history/selection/RNG 等字段，158,430,845 bytes。
- 不上传权重、原始数据或大型预测到 Git，没有删旧基线或停止他人任务。

## 完成核验

1. SUMMARY complete、history 50轮、正式进程已退出、无错误。
2. 八个 BEST checkpoint SHA256 均匹配报告。
3. 八个选点预测文件 SHA256、mask hash、checkpoint/预测 epoch 均匹配。
4. 每率 2610 条 test 标签，logits argmax 与保存的预测一致。
5. 从同一保存预测独立重算 ACC/七类 UA/W-F1，与报告一致；不采样新 mask。
6. 八率和高缺失均值独立重算一致，完整训练恢复文件存在。

机器可读原始结果：`FINAL_SUMMARY.json`，与远程 SUMMARY 字节一致，SHA256
`905c7084eb6d67ea4d08ed96fd995db89169157c56532fd2ccd6c116ac2fa141`。

这只是单 seed、Test-oracle 的特征来源诊断；没有独立测试泛化或统计显著性证明。
