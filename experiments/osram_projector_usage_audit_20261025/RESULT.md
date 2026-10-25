# MOSI 模态投影低使用单元检查

INTERNAL DIAGNOSTIC ONLY

## 结果

在本次 checkpoint 和样本上，没有发现符合预先描述的“各相关可见组合下，
激活和梯度均低”的单元。暂不支持据此启动局部单元重初始化训练。
这不否定原论文的整路软重初始化，也不证明没有其他表示优化空间。

| 投影 | 隐藏宽度 | 1%阈值 | 5%阈值 | 10%阈值 | 各条件全程精确零激活单元 |
|---|---:|---:|---:|---:|---:|
| Audio | 256 | 0 | 0 | 0 | 0 |
| Text | 256 | 0 | 0 | 0 | 0 |
| Visual | 256 | 0 | 0 | 0 | 0 |

阈值：每个条件内，单元平均绝对激活/平均绝对梯度分别除以该条件通道均值，
二者均小于阈值，且在包含该模态的四个条件下全部满足。阈值仅为描述性统计，
不是经过验证的重置规则。单独条件、单独rate有足够支持的统计也没有10%候选。

## 样本支持

| 当前组合 | 八率累计话语槽位数 |
|---|---:|
| A | 1398 |
| T | 1267 |
| V | 1310 |
| AT | 909 |
| AV | 894 |
| TV | 968 |
| ATV | 3526 |

同一批52段训练对话、1284条话语在八个rate重复，共10272话语槽位，不能视为
10272条独立样本。原训练loader共两batch（32+20段），实际每rate两batch、
总16batch；脚本上限为四batch。原始JSON limitation中的four batches是上限
措辞错误，实际batches/cohort记录正确，原始证据保留。

## 固定条件与验证

- 原Flat MOSI seed66、gcnet-random-mask-v1、cmerc-repro；同一miss0.3 BEST，epoch75。
- checkpoint SHA256：`5fe31689597ca975ccbc60eeb2d3235eafafe7e64bee6979f8b3acf7a5fa40fe`。
- 权重来源：`/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2/best_miss_0p3.pt`。
- 捕获真实fc2输入；eval禁用Dropout；对有效话语MSE求和，以autograd.grad取得隐藏梯度。
- 梯度包含历史latent通过CHM影响后续话语的作用；分组是源话语的当前availability。
- 不用测试标签，不更新参数，不重置单元，不运行优化器。
- 每个真实batch：带hook与不带hook预测逐位一致；hook前后CPU/CUDA RNG相同。
- 全程参数摘要一致，无参数.grad累积；padding和rate0/rate.7边界均通过。
- 两项统计单元测试通过；代码compileall与git diff检查通过。
- Torch1.13.1+cu117 / NumPy1.26.4 / sklearn1.3.2；biggpu物理GPU0。
- 实测25.11秒（脚本内部，含setup），峰值分配显存1142.21MiB。
- 环境出现已有torch-sparse可选扩展导入警告；未安装依赖，实际原Flat运行完成。

## 追溯

执行源码：`0f19b86`；后来仅修正文档/输出中的batch上限措辞。
快照：`/data2/yb/remote_experiments/osram_projector_usage_audit_20261025/source_0f19b86`。
输出：`/data2/yb/remote_experiments/osram_projector_usage_audit_20261025/execution_v1`。
完整通道/条件/逐率统计、配置、环境、来源hash见`completed/RESULT.json`；
训练mask保留远端，不上传权重或特征。

## 限制

只有一个checkpoint，无法证明持续低使用或恢复可塑性的必要性；
聚合梯度可以有抵消，激活/梯度大小不能直接表示因果贡献；
没有对单元执行消融，因此不能证明每个单元都必需。
下一步不自动训练、不放宽阈值凑出候选、不扩大数据集或种子。
