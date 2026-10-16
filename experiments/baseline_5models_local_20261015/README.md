# HAMR 五种 Baseline：独立代码审查包

本目录包含五种Baseline的八个版本、MELD/IEMOCAP-4/6数据接口、固定测试Mask、训练器、双GPU调度、真实Checkpoint验证和三种子统计代码。**代码可审查，不代表三种子训练已经全部完成。** 此分支只交付代码、协议配置、测试和源码适配说明；不含数据、权重、日志、预测、成绩、运行队列或GPU任务状态。

本次上传由用户另行明确授权。发布目标仅为 `YB123-DT/git_gcnet` 的 `feature/baseline-5models-local-20261015`；本机训练与实验产物仍保持原有禁止上传约束。

## 模型与协议

| 模型版本 | 时间访问 | 适配边界 |
|---|---|---|
| MMIN UTT-adapted Original | Utterance-only Causal | 既有预提取话语特征编码适配，Teacher/ResidualAE/Reconstruction/Cycle/Classifier保留，无历史编码器 |
| MoMKE-Causal | Causal | 仅时间Self-Attention可见性变化，保留Experts/SoftMoE/路由/融合/两阶段 |
| GCNet Original | Non-causal | 双向LSTM、未来窗口与双图保留；MELD为九说话人、81关系的MELD-adapted Original |
| SDR-GNN Released-compatible（主文） | Non-causal | 保留发布代码Tuple选择、MHA轴及重建Padding行为；只做MELD输入维度/整除头数和九说话人关系兼容 |
| SDR-GNN Implementation-fixed（单列比较） | Non-causal | 既有图返回、Attention轴和重建Padding修复版，保留原产物，不能代填主文Released成绩 |
| ComP-Causal | Causal | 复用feature/comp-meld既有因果实现，不重新设计 |
| MoMKE Original | Non-causal | 附录，与Causal共享数据/MR/参数，仅关闭时间因果Mask |
| ComP Original compatibility adaptation | Non-causal | 附录，启用既有非因果Attention与Prompt，保留兼容修复 |

Seed为66/67/68，MR为0.0–0.7；576个Seed配置、2112个正式Fold任务。MMIN只需要33个Teacher身份并跨MR复用。IEMOCAP在每个Seed内先收齐五个留Session折的真实预测，再跨三个Seed计算均值、样本标准差和方差（ddof=1）。不齐Fold或Seed不能标完整。

**所有正式结果采用 `checkpoint_selection=test_best_acc`、`evaluation_protocol=test_oracle`。这是根据测试表现选模的协议，存在乐观选择偏差，不能称为 unbiased、independent final test 或 leakage-free。** ACC相同时保留更早Epoch，WF1/UA/Macro F1全部来自同一Epoch预测；MoMKE/ComP仅Stage2允许正式选模。

训练Mask在线随机；测试Mask由Dataset/Dialogue ID/Utterance ID/MR/固定评估Seed20261015决定，跨模型/训练Seed/Epoch/Batch一致。采样保留至少一个模态，MR0.7实际饱和为2/3；不与历史动态Mask结果混算。

MELD使用ECERC A300/T1024(eroberta1)/V342、七分类、九说话人与原Train/Valid/Test划分。IEMOCAP使用wav2vec-large-c-UTT A512、deberta-large-4-UTT T1024、manet_UTT V1024，五个Session留一，其余四个Session训练，不新增验证集减少样本。输入文件需要本地自行提供，仓库不附数据。

## 代码审查入口

建议先看 [REVIEW_GUIDE.md](REVIEW_GUIDE.md)，其中列出关键文件、科学有效性检查点、已修复边界与继承限制。`configs/frozen_defaults.json`提供24组模型/数据集默认参数，`configure.py`生成完整任务矩阵；运行状态和逐Run配置是本地生成物。

## 环境与CPU验证

现有实测环境为Python3.10、torch2.2.2+cu121、numpy1.24.3、sklearn1.3.2、pandas2.0.3、PyG2.4.0、torch-scatter2.1.2+pt22cu121。依赖说明见requirements.txt；本次交付没有安装新包。

```bash
EXP=experiments/baseline_5models_local_20261015
PYTHON=/home/yangbin/miniconda3/envs/multimodalerc310/bin/python

# 全部CPU单元/回归测试；没有本地数据时跳过3项数据完整性检查。
"$PYTHON" "$EXP/verify_cpu.py"

# 在已有本地特征的机器执行额外完整数据检查，仍不占GPU。
"$PYTHON" "$EXP/verify_cpu.py" --with-data
```

MoMKE Original等价测试所需的两个官方源文件已作为只读参考放在tests/reference_momke，附官方SHA及文件哈希；不需要克隆上游或联网测试。

## 本地运行与隔离

默认产物目录是 **当前checkout** 的 `runs/baseline_5models_local_20261015/`，不是已有本地实验的产物目录。数据缓存、Teacher、权重和日志都随这个隔离目录保存，并由Git忽略。`BASELINE_RUN_ROOT`可指定另一独立产物目录；`PAPER_WORKSPACE`选择只读输入资源；`BASELINE_PYTHON`选择已安装环境。详见 [ISOLATION.md](ISOLATION.md)。

训练入口仍严格限制本机user23和指定两个GPU UUID，每卡最多一个任务、CPU4线程；GPU忙时等待，不终止其他任务，不支持SSH/远程训练。不要从此审查checkout另起GPU队列与现有任务抢卡。

在本机无同资源活动队列、需要独立新运行时才使用：

```bash
export BASELINE_PYTHON=/home/yangbin/miniconda3/envs/multimodalerc310/bin/python
"$BASELINE_PYTHON" "$EXP/configure.py"
"$BASELINE_PYTHON" "$EXP/preflight.py"
"$EXP/dispatch_local_2gpu.sh"
```

预检通过后才打开各模型门控。MoMKE MELD四个预算pilot先比较100/50与200/100：仅用最终10轮Stage2训练loss的预定规则选预算，禁止读取测试分数决策。两版本统一预算，完全兼容的选中pilot显式复用，避免重训。

Checkpoint保存真实模型/优化器/Epoch/RNG快照，完成前重新加载最佳权重回放预测；调度器核验进程、退出码、预测、Mask和文件哈希，不能仅因存在文本文件就认定成功。失败任务保留证据并停止自动重试。

`aggregate_results.py`按真实本地产物生成progress.csv、results.csv、results_seed.csv、results_summary.csv、TABLE_A.md、TABLE_B.md、TABLE_C.csv、TABLE_D.md、sdr_version_comparison.csv和配对比较；这些生成物不随代码上传。无匹配HAMR结果时留空，不将OSRAM结果冒充HAMR。

## SDR-GNN发布代码与修复版的边界

主文Table A只接收`released_compatible`；`implementation_fixed`保留为独立版本，在Table D及`sdr_version_comparison.csv`按相同样本/固定Mask配对。Released未完成时主文留空，不能借用Fixed成绩。此处仅交付代码，不附实验成绩或运行状态。

Released只允许两类数值兼容：MELD A+T+V=1666，Attention采用不超过原256头的最大整除头数238（每头7维）；九说话人对应81种Speaker关系，按`9*i+j`编号。IEMOCAP保持2560/256与原两说话人Set关系逻辑，启动Python前设置冻结的`PYTHONHASHSEED`。包导入改用相对路径以隔离模块。

原Tuple拼接后只取Temporal分类/重建、MHA沿对话Batch轴注意力、重建T/B与Padding展平次序均保留。这些行为不是本分支对Released算法的修复；完整Diff和来源见`adapters/graph_models/sdrgnn_released/`。新队列的preflight覆盖两版本；已有队列扩展工具`extend_sdr_released_queue.py`只接受本checkout处于暂停状态的Dispatcher、同一`BASELINE_RUN_ROOT`及与当前源码哈希匹配的本地CPU验证记录，保留已有任务配置。扩展工具不会自动停止训练、运行GPU或上传产物，不要对活动实验盲目执行。
