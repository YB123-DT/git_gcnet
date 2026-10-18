# MELD：每个缺失率独立训练、同缺失率测试

INTERNAL SINGLE-RATE TEST-ORACLE SCREEN

八组已全部完成50/50轮；最终逐率 ACC/UA/W-F1、选中 epoch 和比较边界见
RESULT.md，精确指标及选点产物 hash 见 FINAL_SUMMARY.json。

用户授权一轮 MELD 固定率训练。默认解释为 ECERC 原 Flat、seed66；本轮不延续
MMGCN 特征，也不引入余弦、模块或其他调参。当前仅授权下列 8 组，不扩展种子。

| Job | 训练 rate | 测试 rate | 模型 |
|---|---:|---:|---|
| R0 | .0 | .0 | 独立从头训练 |
| R1 | .1 | .1 | 独立从头训练 |
| R2 | .2 | .2 | 独立从头训练 |
| R3 | .3 | .3 | 独立从头训练 |
| R4 | .4 | .4 | 独立从头训练 |
| R5 | .5 | .5 | 独立从头训练 |
| R6 | .6 | .6 | 独立从头训练 |
| R7 | .7 | .7 | 独立从头训练 |

**固定的是缺失率，不是固定一份 mask，更不是 persistent whole-modality missing。**
原 GCNet 的话语级随机 mask 仍按原实现生成；Host/Guest 分开采样、padding
清零、有效话语至少一个模态；.7 对应原实现的单模态保留（实际缺失比例 2/3）。

## 固定配置

复用 `experiments/osram_meld_mmgcn_features_20261018/CONFIG.json` 中的原 Flat
训练配置，但数据明确使用 ECERC，而不是 MMGCN：

- `/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl`。
- SHA256 `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。
- train/dev/test 保持原划分及全部样本。
- MELD、seed66、50 epoch、batch32、Adam LR=.001 constant、weight decay=1e-5、dropout=.5。
- emotion-only/no-JEPA；原 Flat1600、Local Skip、8 heads、K/V64、write_step=.6。
- CHM、Query、联合 Block Delta Write、causal read-before-write、损失及分类头不变。
- `missing_mask_protocol=gcnet-random-mask-v1`，不得切换 conversation-v2 或 legacy-fixed。
- 每组只改 `train_rate_mode=fixed`、`fixed_missing_rate=r`、`train_missing_rates=[r]`。
- 每组独立训练、优化器和完整恢复状态独立，初始化 hash 应相同，不加载 cyclic 的权重。

## 最小训练器兼容修改

原 `_protocol_rates` 和 `train_epoch` 已有 fixed 分支，实际 mask 算法无需修改。
本次只放宽两个入口限制：

1. 显式 fixed 配置可以使用 `test-oracle-per-rate` 的选点保存路径；此时
   protocol_rates 只有一个 rate，不会偷偷评测其他七率。
2. 完整 epoch-boundary `TrainingState` 允许 fixed emotion-only 模式。

原 cyclic 默认行为、计算公式与 RNG 调用不变。旧固定率被拒绝的契约测试相应
更新，并新增同率测试循环、保持 cyclic 八率和完整恢复入口的回归检查。

## 选点和产物

每个模型按自身同一 rate 的 Test ACC 选择 best；保存该次 forward 的原始
logits/labels/predictions/availability/话语身份及 checkpoint、mask hash。
UA 是选中 ACC 的同一批预测的七类 Macro Recall，W-F1 同理，不从训练结束后
新随机抽样的测试结果补数。

每组保留自身 BEST、选点预测、完整 last_training.pt、history、INITIAL_STATE、
EPOCH_STATUS、effective config、SOURCE_MANIFEST hash、环境和 GPU UUID。
仅清理该新 run 中没有被任何 BEST/last 引用的临时训练版本，不删旧结果。
resume 要求完整状态、同源/同配置/同数据，不把只加载 BEST 称为完整续训。

EPOCH_STATUS 会逐轮核对：33 个 optimizer steps、1038 段训练对话、全部 batch
只使用该 rate、测试也只包含该 rate、JEPA loss=0；参数量须为原 13,290,795。
实际第一轮属于正式预算，是集成检查，不另加独立 smoke 训练条件。

## 有限并发队列

- 服务器 biggpu，原 `/data2/yb/conda_envs/cmerc-repro/bin/python`，不改变环境。
- 物理 GPU0：`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`。
- 物理 GPU5：`GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`。
- 每卡最多4组，总共8组；通过 UUID 绑定，进程内 cuda:0；GPU4 禁止。
- R0 提交真实第一轮后自动放行余下7组，预计达到每卡4组，不把未启动组说成运行中。
- 复用已有持久 Scheduler，启动前重查显存和磁盘、输出独立、失败记录，OOM 最多
  一次降低并发重试且不改超参；不杀、暂停或改动其他人的任务。
- 队列只跑完本轮8组，自动汇总后停止；不追加种子、数据或搜索。

从固定的源码快照运行（根目录必须有匹配的 `SOURCE_MANIFEST.json`）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_fixed_rates_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_fixed_rates_20261018/runs_v1
```

SSH 断开后由 tmux 持续运行。实际源 commit、部署路径、PID、进度与日志在
`LAUNCH.md` 记录；本 README 只是入口说明，不声称已经启动或完成。

## 比较边界

这是 **8个 rate-specific 模型**，不是一个模型覆盖八率。每个模型各50轮，
总计400个 model-epochs、8×1650个 optimizer step；原 cyclic 单模型只有50轮。
每个 rate 获得更多专用训练曝光，不能把改善直接归为结构创新或等计算量收益。

同 seed 不代表与旧 cyclic 结果具有相同实际训练/测试 mask：调用次数与训练
分布都改变，选中 epoch 也可能不同。主结果只描述各率专用训练的表现，保留
已有 cyclic seed66 的同口径结果作历史参考，不伪称 exact-mask 配对。

## 检查

新测试与原 per-rate/选点回归测试已验证固定率入口、单率评测、原 cyclic 八率
及原 exact-selection 口径。后续仍需实际首轮/完整训练产物证明运行成功。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q --tb=short \
  tests/test_meld_fixed_rate_training.py tests/test_osram_per_rate_selection.py \
  tests/test_meld_block2_selection.py
```

本地检查环境不是训练环境迁移。子agent测试入口不可用，因此上述测试由主线
实际执行；不把失败的子agent调用算作验证完成。既有 optional torch-sparse
导入 warning 与此前相同；是否影响实际训练由真实首轮检查确认。
