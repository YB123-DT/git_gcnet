# GCNet 第二图层机制实验记录

详细主协议：[EXPERIMENT.md](EXPERIMENT.md)。英文版：[EXPERIMENT.en.md](EXPERIMENT.en.md)。

## 当前边界

这份文件记录预注册方案，不把“代码写完”冒充“效果变好”。截至 2026-08-25：

| 候选 | 精确插入位置 | 参数变化 | 状态 |
|---|---|---:|---|
| GenAgg | 只替换 temporal、speaker 两支的第二层 `GraphConv` 聚合 | 总计 +118 | 已实现；Phase A 正式训练待运行 |
| Scaled Soft Medoid | 同上 | +0 | 已实现；Phase A 正式训练待运行 |
| SSMA Conv2 | 同上 | 每支 +297,700，总计 +595,400 | 已实现；官方环境门禁通过；Phase B 正式训练待运行 |
| 自定义 RTDR | 计划只改变二跳 relation-transition 路由 | 预计 +0，尚未验证 | 实现、等价测试、训练均 `PENDING` |

GenAgg 的理由是普通 sum 不能学习邻居数量、中心化与非线性集合行为；Soft Medoid 检验少数消息空间异常点是否主导聚合；SSMA 在压缩前显式产生跨邻居乘积；RTDR 检验删去 `q≠r` 的二跳 relation transition 是否有用。RTDR 不能写成 MrMP/MMP 迁移，因为检查过的 [MrMP 论文](https://arxiv.org/abs/2202.04844)在每一层内部就会把 relation 消息求和。

可追溯来源：[GenAgg 论文](https://arxiv.org/abs/2306.13826)及[论文时期代码](https://github.com/Acciorocketships/generalised-aggregation/blob/3c95c10afac4bda77afc30e80a7481c7e537fca1/genagg/genagg.py)，[Soft Medoid 论文](https://arxiv.org/abs/2010.15651)及[官方代码](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/blob/4f94140afb7fd2ef5bf77f45a5efc7b2d6eb2a09/rgnn/means.py)，[SSMA NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html)及[检查的代码 commit](https://github.com/AlmogDavid/SSMA/tree/9d128c902acf47343b5baf5150a78dfb6a64fb3e)。完整公式、来源边界和否决依据见[主协议](EXPERIMENT.md)与[新增候选决策](../../docs/superpowers/specs/2026-08-25-additional-graph-candidates-design.md)。

## SSMA 官方环境事实

SSMA 核心 commit 为 `08aa55fb255d5e32aa9f6171246e6e2821c97c71`。在 biggpu 的正式环境 Python 3.8.20、Torch 1.8.0、PyG 2.0.1 中，CPU/GPU FP32 forward 与 backward 均 finite；实测每支新增参数 297,700。64 节点 GPU 合成检查峰值为 allocated 56,989,696 bytes、reserved 92,274,688 bytes。Torch 1.8 出现 complex-to-real 警告，但梯度仍 finite。这个结果只证明兼容性与合成显存范围，不是 epoch smoke，更不是 IEMOCAPSix 精度结果。

## 两阶段正式任务

- Phase A：`{genagg, soft_medoid}` × missing `{0.0,0.7}` × seeds `{66,67,68}`，共 12 个任务。
- Phase B：`{ssma,rtdr}` × 同样的 rate/seed，共 12 个任务；RTDR 通过 Original 原路径 bit-exact 与 full-transition 等价测试后才能启动。
- 两阶段都固定 `stage=formal`、fold 5、四张卡、每卡三个进程，总计 24 个新候选任务，Original 任务数为 0。

Original 的 40 个 NPZ 从以下只读目录继承，不重新训练：

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

未来配对键为 missing rate、seed、fold 与 mask SHA256。在启动候选训练或做任何结果比较前，必须先实现、测试并通过专用的 Original legacy-aware validator。该未来门禁只允许把旧 Original 中历史上不存在的字段映射到锁定默认值 `addition/bilstm/bilstm/add`；source、命令、数据集、特征、参数数、rate、seed、fold 和 mask hash 仍必须严格一致。这里规定的是尚待完成的强制门禁，不表示 validator 已经实现。

## 效率约束与判定

不跑 epoch smoke；每个候选只做一次正式环境合成前后向。每个 Phase 一次启动 12 个任务，复用已有 runner 的 manifest、锁、task key 与 resume，已经完整的任务不重跑。任务必须满足 return code 0、100 epoch、恰好一个可读 NPZ、指标 finite、代码/命令/mask provenance 一致才算完成。科学上没有提升的完整实验要作为负结果保留；半写、损坏或 provenance 不一致的任务只能标 `INCOMPLETE`。

候选晋级要求：两个 rate 的配对均值增量都为正、seed-macro 配对增量为正、三个 seed 中至少两个为正，并且没有 non-finite 或坍塌。SSMA 若为正，还必须增加 parameter-matched sum+MLP 控制，才能把收益归因于邻居交互而不是参数量。

## 跑完后自动上传 GitHub

正式训练完成后不再等待用户发命令。自动流程会：

1. 等待两波任务到终态并完成 Original 配对与 provenance 校验；
2. 将代码、任务级 NPZ、必要日志、run/invocation manifest、源文件与 mask hash、`RESULTS.md/.zh.md/.en.md` 同步回 `/data2/yb/paper`；
3. 在将要发布的树上重跑测试和结果校验，按 Lore 协议提交；
4. 推送到 [YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet) 的 `exp/second-graph-aggregators` 分支，并用 `git ls-remote` 核对远端 SHA；
5. 推送失败只从同一个本地 commit 重试，绝不因此重跑训练；合入 GitHub `main` 的完成版本目录时不 force-push，也不改写其他已完成版本。

数据集、巨大特征文件、登录凭据、缓存和未完成产物不会上传。通过 provenance 的科学负结果会如实上传；基础设施未完成的目录不会包装成“完成版本”。当前正式指标与 RTDR 仍为 **PENDING**，本文没有虚构任何 F1 结果。
