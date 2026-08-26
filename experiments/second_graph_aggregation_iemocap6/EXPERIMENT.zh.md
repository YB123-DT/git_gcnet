# GCNet 第二图层机制实验记录

详细主协议：[EXPERIMENT.md](EXPERIMENT.md)。英文版：[EXPERIMENT.en.md](EXPERIMENT.en.md)。

## 当前边界

这份文件按时间顺序保留四层证据，不追溯改写早期判定：四个模块先后经历初始6任务 gate、RTDR 15对扩展、RTDR 40对 full-rate audit，最后全部放到统一的三 missing rate、五 seed 网格。Original 全程没有重新训练。详见[中文 RESULTS](results/RESULTS.zh.md)、[中文 ANALYSIS](results/ANALYSIS.zh.md)和[统一逐任务证据](results/uniform_three_rate/RESULTS.zh.md)。

| 候选 | 精确插入位置 | 参数变化 | 状态 |
|---|---|---:|---|
| GenAgg | 只替换 temporal、speaker 两支的第二层 `GraphConv` 聚合 | 总计 +118 | 已实现；6/6 success；gate FAIL（`-0.187831724`，发生坍塌） |
| Scaled Soft Medoid | 同上 | +0 | 已实现；6/6 success；gate FAIL（`-0.004304363`） |
| SSMA Conv2 | 同上 | 每支 +297,700，总计 +595,400 | 已实现；6/6 success；gate FAIL（`-0.007173215`） |
| 自定义 RTDR | 只改变二跳 relation-transition 路由 | 已验证 +0 | 初始6/6 gate FAIL（`+0.002541466`，1/3 seeds为正）；15/15 extension限定判据PASS；40/40 full audit `stable_positive=false`（overall `-0.002810103`，3/8 rates为正，3/5 seeds为正） |

GenAgg 的理由是普通 sum 不能学习邻居数量、中心化与非线性集合行为；Soft Medoid 检验少数消息空间异常点是否主导聚合；SSMA 在压缩前显式产生跨邻居乘积；RTDR 检验删去 `q≠r` 的二跳 relation transition 是否有用。RTDR 不能写成 MrMP/MMP 迁移，因为检查过的 [MrMP 论文](https://arxiv.org/abs/2202.04844)在每一层内部就会把 relation 消息求和。RTDR 核心 commit `8f375b2509016daf5395863b0220591bc8bcd3ee`、CLI/runner commit `a107f7448978f4c22f87a6b61ec45b53da312aa0`；新增参数为零，官方核心检查中 untouched early path 误差为 0，full-transition 分解误差为 `5.96e-8`。

可追溯来源：[GenAgg 论文](https://arxiv.org/abs/2306.13826)及[论文时期代码](https://github.com/Acciorocketships/generalised-aggregation/blob/3c95c10afac4bda77afc30e80a7481c7e537fca1/genagg/genagg.py)，[Soft Medoid 论文](https://arxiv.org/abs/2010.15651)及[官方代码](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/blob/4f94140afb7fd2ef5bf77f45a5efc7b2d6eb2a09/rgnn/means.py)，[SSMA NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html)及[检查的代码 commit](https://github.com/AlmogDavid/SSMA/tree/9d128c902acf47343b5baf5150a78dfb6a64fb3e)。完整公式、来源边界和否决依据见[主协议](EXPERIMENT.md)与[新增候选决策](../../docs/superpowers/specs/2026-08-25-additional-graph-candidates-design.md)。

## SSMA 官方环境事实

SSMA 核心 commit 为 `08aa55fb255d5e32aa9f6171246e6e2821c97c71`。在 biggpu 的正式环境 Python 3.8.20、Torch 1.8.0、PyG 2.0.1 中，CPU/GPU FP32 forward 与 backward 均 finite；实测每支新增参数 297,700。64 节点 GPU 合成检查峰值为 allocated 56,989,696 bytes、reserved 92,274,688 bytes。Torch 1.8 出现 complex-to-real 警告，但梯度仍 finite。这个结果只证明兼容性与合成显存范围，不是 epoch smoke，更不是 IEMOCAPSix 精度结果。

## 两阶段正式任务

- Phase A：`{genagg, soft_medoid}` × missing `{0.0,0.7}` × seeds `{66,67,68}`，共 12 个任务。
- Phase B：`{ssma,rtdr}` × 同样的 rate/seed，共 12 个任务。
- 两阶段都固定 `stage=formal`、fold 5，总计 24/24 个新候选任务 success，Original 任务数为 0。

Phase A 初始使用 GPU 4–7、每卡三个任务；GPU 4 的正式训练进程连续以 `-9` 退出，这些失败 attempt 只保留为 diagnostics，不计入 canonical 结果。对应三个 canonical task key 随后在 GPU 5、6、7 重跑并成功。Phase B 使用 GPU 5–7、每卡三个并发，空闲后队列自动接续，最终 12/12 success。

### RTDR post-gate 稳定性审计

初始 gate 保持不变。看到初始结果后，用户明确要求继续检查 RTDR 的稳定性。15对 extension 覆盖 missing `{0.0,0.5,0.7}` 与 seeds `{66,67,68,69,70}`，复用已有6个任务，只新增9次训练；其单独锁定的扩展判据PASS，overall为 `+0.008510981`，3个rate均值均为正，3/5 seed macros为正。随后 full audit 覆盖8个missing rate×5 seeds，复用这15个任务，只新增25次训练，最终形成40/40个 provenance-valid RTDR 归档。

Full audit 的预定义条件是：overall macro delta为正、至少6/8个rate均值为正、至少3/5个seed macro为正，且全部运行finite、六类不坍塌。实际 `stable_positive=false`：overall为 `-0.002810103`，仅3/8个rate均值为正，3/5个seed macros为正。

## 统一三档证据

共同网格为 `{0.0,0.5,0.7}` × seeds `{66,67,68,69,70}`，每臂 15 对。GenAgg、Soft Medoid、SSMA 各新训练 9 次，共新增 27 次；RTDR 复用已有 15 个单元。正式 invocation 使用 GPU 1–7；GPU 4 三次 `-9` attempt 移入 diagnostics 后，相同 task identity 在 GPU 1–3 成功完成。

GenAgg `-0.204847963`、0/3 rates、0/5 seeds，且有坍塌（`uniform_stable=false`）；Soft Medoid `+0.004706753`、2/3 rates、4/5 seeds、无坍塌，但 `0.7` 为 `-0.002089281`（`false`）；SSMA `-0.001153174`、1/3 rates、2/5 seeds（`false`）；RTDR `+0.008510981`、3/3 rates、3/5 seeds（`true`），但 full-rate `stable_positive=false` 仍然成立。最终唯一 candidate 归档数为 85 = 15 + 15 + 15 + 40，Original 重训练数为 0。

Original 的 40 个 NPZ 从以下只读目录继承，不重新训练：

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

配对键为 missing rate、seed、fold 与 mask SHA256。commit `d515386f3207105c8207c34eca3f9743d2b80e4f` 已实现 fail-closed Original legacy-aware validator。最终唯一 candidate 归档数为 85，Original 训练数全程为 0。

## 效率约束与判定

没有跑 epoch smoke；每个候选只做正式环境合成前后向门禁。两波均复用已有 runner 的 manifest、锁、task key 与 resume；已经完整的 canonical task 不重跑。初始24个正式任务以及后续新增的RTDR任务均满足 return code 0、100 epoch、唯一可读 NPZ、指标 finite 和代码/命令/mask provenance 一致。

初始候选晋级要求是两个 rate 的配对均值增量都为正、seed-macro 配对增量为正、三个 seed 中至少两个为正，并且没有 non-finite 或坍塌。真实结果为：GenAgg `-0.187831724` 且发生坍塌；Soft Medoid `-0.004304363`；SSMA `-0.007173215`；RTDR 虽为 `+0.002541466`，但仅 1/3 seed macros 为正。因此按初始预注册规则，四者 gate 均 FAIL，没有候选获得扩展资格。后续用户要求的 RTDR 审计单独报告，既不改写初始FAIL，也没有通过full-audit `stable_positive`。

## 跑完后自动上传 GitHub

早期58个candidate归档的证据 commit `97370fd49cb130bc10c620f1293ebff00985b729` 不包含本轮新增27次训练。当前85归档统一证据层已通过360/360项哈希和267项测试（1项预期跳过），并作为证据 commit `f72cd776d5260644a84005241e55639b994bb1dc` 推送；`git ls-remote` 返回相同SHA。自动流程为：

1. 当时发布的 58/58 candidate 快照与 Original 40/40 属于早期证据层；当前统一层将唯一 candidate 归档数扩展到 85，不借用早期 commit 声称已发布；
2. 保留代码、任务级 NPZ、必要日志、run/invocation manifest、源文件与 mask hash，以及初始 `RESULTS.md/.zh.md/.en.md`、`ANALYSIS.md/.zh.md/.en.md`和RTDR extension/full汇总；
3. 在将要发布的树上重跑测试和结果校验，按 Lore 协议提交；
4. 推送到 [YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet) 的 `exp/second-graph-aggregators` 分支，并用 `git ls-remote` 核对远端 SHA；
5. 推送失败只从同一个本地 commit 重试，绝不因此重跑训练；合入 GitHub `main` 的完成版本目录时不 force-push，也不改写其他已完成版本。

数据集、巨大特征文件、登录凭据、缓存和 diagnostics-only 失败 attempt 不作为 canonical 结果上传。早期已发布证据保留初始四臂负 gate、RTDR 限定 extension PASS 和 full-audit `stable_positive=false`；当前第四层统一证据将在完成本轮验证后另行发布，不借用旧SHA声称已经上传。
