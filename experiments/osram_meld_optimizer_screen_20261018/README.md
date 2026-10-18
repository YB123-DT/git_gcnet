# MELD 原 Flat：较小 LR / 较大 weight decay 的有限筛选

INTERNAL TEST-ORACLE SCREEN。不是 IEMOCAP Probe 调参，也不是加入第二 Block。

用户明确授权重新训练 MELD Flat，并在空闲卡多开几组。本轮限定8组 seed66，
全部从头训练；不扩展种子、不自动接 Optuna、不改正在执行的 IEMOCAP Probe。

| Job | learning_rate | weight_decay |
|---|---:|---:|
| O0 | 2e-4 | 2e-4 |
| O1 | 2e-4 | 1e-3 |
| O2 | 1e-4 | 2e-4 |
| O3 | 1e-4 | 1e-3 |
| O4 | 3e-5 | 2e-4 |
| O5 | 3e-5 | 1e-3 |
| O6 | **1e-5** | **2e-4** |
| O7 | 1e-5 | 1e-3 |

仅改变原有效配置的 `learning_rate` 和 `weight_decay`。使用原 Adam，而非
擅自改为 AdamW；因此这里的 weight decay 仍采用原优化器语义。

## 完全不变的协议

- 原 MELD ECERC 特征及官方 train/dev/test。
- 特征 SHA256：`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。
- 原 Flat1600 + Local Skip，8 heads / K-V64，write_step=.6，causal
  read-before-write、真实观测联合 Block Delta Write。
- no-JEPA、emotion-only，原任务头与CE，不加任何新模块或辅助损失。
- 原 `gcnet-random-mask-v1`，batch32、dropout=.5、cyclic0.0–0.7。
- **50轮、constant LR**；不同时引入 cosine/warmup 等新变量。
- 每率按 Test ACC 选 BEST；UA/W-F1/CE/类别 Recall 均从该次原始预测计算，
  不从最后一次重新采样测试补数，也不把 UA-best 混到 ACC-best 表内。

## 对照及可比性

只读复用已经完成的同源码、同环境 constant-LR W0，不重跑相同对照：

```text
/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0
```

W0 seed66 八率 ACC60.838123 / 对应UA35.632827 / W-F1 56.788705；
其 LR=.001、weight_decay=1e-5。不要将单种子调参分数与历史三种子均值混比，
也不据此自动替换用户接受的 Flat baseline。

原数值源码固定：`b4a313841f098487a6482d2712bdac94aa8e319e`，位置：

```text
/data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138
```

复用原训练器、TrainingCapture、原子 TrainingState 和选点保存，不修改主干
源码。新增观察器每轮核对实际 optimizer 参数组的 LR/WD 和原 W0 的实际
train/test mask stream（含hash、batch数、有效样本数）；仅same seed不算配对。
旧训练器没有参数组name，按真实参数归属给组的副本加审计名称，不修改Optimizer。

所有初始化hash须与W0相同，参数量须为13,290,795。尽管各轮掩码一致，最终
各rate选中epoch仍可能不同，所以不声称选点预测也恰好使用相同mask。

小 LR 在50轮可能尚未充分收敛；本轮预算不变，不能把它的低分直接解释为
参数永远无效。查看最优epoch是否挤到预算尾部，但不自动延长训练。

## 环境、资源及执行

只在 biggpu 原 `/data2/yb/conda_envs/cmerc-repro/bin/python` 运行，严格检查
Torch1.13.1+cu117、NumPy1.26.4、sklearn1.3.2，不换环境。

使用健康物理GPU1/2，每卡最多4组，总共8组：

- GPU1：`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`。
- GPU2：`GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`。

GPU4禁止；GPU0/5的IEMOCAP Probe、GPU6/7原有任务不动。每个新任务使用
独立输出目录；复用原Flat的真实显存测量作准入依据，每任务保守预留4GiB，
启动前重查显存。只在有限OOM重试时降低并发，不偷偷改batch或超参。

不可变实验快照须包含根目录 SOURCE_MANIFEST.json；源代码不随本地后续编辑
变化。用tmux持久执行并自动完成八组及汇总：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_optimizer_screen_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1
```

现有输出恢复必须显式 `--resume` 且源、配置、数据、对照hash相同；保留完整
last_training.pt（模型/optimizer/RNG/schedule/实际LR记录）与每率BEST和精确
选点预测。只清理本新run内部无引用的临时训练版本，不删受保护旧权重。

实际启动信息见 LAUNCH.md；实时查 RUNNER.json、queue/QUEUE.json、各run
EPOCH_STATUS.json。全部完成后生成 RESULTS.json 和 RESULT.md，包含失败状态。

## CPU检查

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q \
  tests/test_meld_optimizer_screen.py tests/test_meld_write_tuning_run.py \
  tests/test_meld_write_tuning_capture.py
```

本地CPU测试不意味着训练环境迁移。正式评分须等真实50轮产物；多组测试选优
存在选择偏差，不可宣称独立测试泛化提升。训练代码成功不等于方法有效。
