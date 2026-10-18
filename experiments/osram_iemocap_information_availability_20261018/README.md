# IEMOCAP：Local 与 Memory 的冻结表征检验

INTERNAL DIAGNOSTIC ONLY。用户授权复用 MELD 的检验方式，不重训 OSRAM。

## 范围

- IEMOCAP-4 和 IEMOCAP-6，完整 5 折，原 Flat 主干 seed66。
- 八个缺失率，每率使用原先按 Test ACC 选择的 BEST checkpoint。
- 新 `gcnet-random-mask-v1`；不使用旧 legacy-fixed 或 conversation-v2。
- 不改 Local Skip、Flat、Memory、Query、读写、损失，也不加新 Block。
- 只训练离线 Probe，每个 30 轮，初始化 seeds66/67/68；这不是三个主干种子。
- 完全沿用旧 MELD 的 13 种输入：Local；Base/Memory/LocalBase/LocalMemory/
  RawHistory/LocalRawHistory，以及后六种的不同对话 Donor 对照。
- 总计 10 个折任务、80 个 rate/fold 单元、3120 个小头拟合，不是3120次主干训练。

## 表征与对照

冻结提取 Local256、Base512、Gap-A/T/V 各512，仅使用实际有效的 forward half。
Gap 固定槽位，当前可见模态的 Gap 安全置零；首句历史必须为零。简单历史统计
严格使用过去 Local，不含当前或未来句。历史标签不作为任何 Probe 输入。

测试端必须逐行重放 BEST 对应的 selected mask，先校验 conversation ID、句序、
speaker、labels、预测类别及 logits，再按对话排序进行严格历史统计。训练端用
原 GCNet host/guest 分开采样的 padded rectangle，固定一个 rate-specific 离线
视图，所有 Probe 共享。这不是重放完整 cyclic 主干训练，而是诊断头的固定输入。

Donor 来自同 split 的其他对话，匹配当前 availability 和有无历史/同 speaker 历史，
不用情绪标签匹配。每个 Probe 初始化下所有13种输入使用同一 donor-included
训练/测试 cohort。相较旧 MELD 中 Local 使用全量样本，这是显式对齐修正，不能
将两个数据集的数值差异直接解释为模型机制差异。

预测头沿用 `LayerNorm -> Linear(64) -> GELU -> Linear(C)`，C 显式为4或6。
Adam LR=.001、weight decay=1e-5、batch256、train-only 标准化，Test ACC/W-F1/CE
依次决定最佳轮，沿用旧 Probe 规则。UA、W-F1、CE、类别 Recall 均从同批预测
计算，UA 使用全部4/6类。主干选择同样是 Test-oracle，因此仅作内部诊断。

回答：LocalMemory 是否超过 Local？真实 Memory 是否超过同宽 Donor？纠错是否
超过致错？同时报告八率均值、高缺失均值、逐率、逐折与逐 Probe 初始化结果。
先五折等权，再八率等权，最后对三个 Probe 初始化给出均值和样本标准差。
纠错/致错先合并五个互斥测试折，再平均 rate/初始化；重复出现的句子不是新样本。

## 源码、环境与资源

主干数值代码固定原训练快照 `785a01242ef1a82b97d5337acbb33ba963df5537`：

```text
/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012
```

原权重只读复用，80个 BEST 和精确选点预测已核对存在：

```text
/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/runs/iemocap{4,6}/seed_66/fold_{1..5}
```

新实验代码单独冻结快照；worker 检查 SOURCE_MANIFEST，数值 package 绑定原快照，
不让当前 worktree 新功能替代原训练代码。状态 hash 在提取前后和 Probe 训练后
必须一致，主干参数没有梯度。旧 MELD `fit_probe` 只新增可选 `classes=7`，默认
七类路径不变；不复用旧 MELD 的硬编码类别验证器。

环境严格沿用 biggpu `/data2/yb/conda_envs/cmerc-repro/bin/python`：Torch1.13.1+cu117、
NumPy1.26.4、sklearn1.3.2。只用健康 GPU0/5，每卡至多2个折任务；GPU4 禁止，
不改动 GPU6/7 原有任务。已有 Flat/B6 显存记录作保守准入依据，不新增主干训练
smoke。首个真实提取与小头拟合完成后检查真实内存及协议；只完成这有限批次。

## 执行与恢复

从部署后的不可变实验快照运行，需根目录包含匹配的 SOURCE_MANIFEST.json：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_iemocap_information_availability_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_iemocap_information_availability_20261018/runs_v1
```

使用 tmux 持久运行；实际 commit、PID、进度和日志另记 LAUNCH.md。相同输出
恢复必须显式 `--resume`，不得重复启动。已完成 rate/Probe 和特征缓存可复用；
本实验不声称可逐 optimizer-step 恢复一个未完成的小头拟合。

每个 Probe 保存 best.pt、predictions.npz、IDENTITY.json、训练曲线和指标。
每 rate 保存冻结特征、mask来源hash、PROBES.json；每折保存进度 SUMMARY.json。
全批完成后自动生成 FINAL_SUMMARY.json、RESULT.md；失败明确记录，不补造分数。
缓存、预测与权重不上传 GitHub，代码/文本/小型汇总上传。

## 解释边界

Local 与联合输入头的参数量不同；真实/Donor 才是严格同宽容量对照。Probe 的
纠错不代表原 Flat 同一条路径的因果贡献，真实历史优于 Donor 也不能单独定位
写入/读取/决策瓶颈。若 LocalMemory 不优于 Local，必须如实报告，不自动加模块。

## 验证命令

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q \
  tests/test_iemocap_information_probe.py tests/test_iemocap_probe_analysis.py \
  tests/test_meld_information_features.py tests/test_meld_information_probe.py \
  tests/test_meld_information_runner.py tests/test_meld_information_analysis.py
```

本地只做 CPU 测试，不是迁移训练环境；实际提取/Probe 在 biggpu 原环境执行。
