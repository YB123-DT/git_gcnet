# P2 历史训练对照

本目录提供两个从头训练的独立对照，尚未启动正式训练。

- `local-only`：复用既有 `osram_ablation=local-only`，保留 ObservedSet mean、Local、Flat 和 classifier；CHM 仍执行但其 Base/Gap 输出不进入分类器。
- `causal-gru`：显式 `osram_causal_gru_history=True`，使用同一 ObservedSet mean node 输入单层单向 GRU（hidden=256），每步先读取 `h[t-1]`，再更新当前 node。无偏置线性映射到 Base forward 512；Base backward 512 与全部 Gap 为零，保留原 Local+Flat+classifier。CHM 参数在构造后移除，CHM scan/query/write 从不执行。

默认开关为关闭，旧参数键、初始化 RNG 和输出路径保持兼容；GRU 对照是新架构，不能加载 CHM checkpoint 作为同一模型续训。配置拒绝 backbone/joint-pretrain/B2/teacher/text-subspace 的全部预训练来源、预训练 learning rate 和 pretrained-frozen teacher，确保普通 GRU 对照从头训练。GRU 状态在每个 forward 清零，各 batch 列独立；padding 不读取、不更新、不影响有效输出。Flat 最后一层沿用零初始化，因此 GRU 在第一次 task optimizer 更新后才收到非零 task 梯度。

两组固定 cfg84：emotion-only、disable-unused-aux、output=1600、heads=8、key/value=64、write_step=.6、causal、Flat、meaningful=none。保留 batch=32、Adam lr=.001、weight_decay=.00001、cyclic 全部既有缺失率和官方 split。IEMOCAPFour/IEMOCAPSix 均使用 ECERC 已转换 features、五会话 fold、100 epochs、`conversation-mask-v2`；标签来源分别读取既有 `config.PATH_TO_LABEL` 的 `IEMOCAP_features_raw_4way.pkl`/`IEMOCAP_features_raw_6way.pkl`，分类器类别数沿训练器 `_dataset_shape` 的4/6类契约。两组使用同一 utterance 对齐的 ECERC audio/text/visual 根目录，不从数据集名称猜测新特征路径。MELD 使用 ECERC 官方 pickle、fold1、50 epochs、`gcnet-legacy-fixed-v1`。后者直接复用冻结的 `LegacyFixedMaskSchedule`，包括其 host/guest 共用随机键、训练按 epoch 变化和测试固定 epoch0 的行为，没有改称或切换为 conversation-mask-v2。

完整 P2 队列按三个 seeds 生成：IEMOCAPFour 30 个、IEMOCAPSix 30 个、MELD 6 个，共66个独立任务；每个 IEMOCAP 数据集单seed为10个（两个control×五fold）。

P2 显式设置 `checkpoint_selection_metric=weighted_f1` 与 `test-oracle-per-rate`。这是既定探索对照协议，会用测试集选每个缺失率的最佳 epoch；不能作为严格独立测试集泛化证据。其他调用仍默认 `dataset-default`（现有情感任务按 accuracy 选点）。每个 run 保存有效 config、数据 hash、feature manifest、commit/差异、环境、日志、预测、权重和 provenance。

部署为不含 `.git` 的 `git archive` 快照时，provenance 不调用 Git，改为记录快照目录身份、P2 核心文件及 runner/冻结 mask schedule 的 SHA256 和组合身份。可显式设置 `OSRAM_P2_SOURCE_COMMIT` 关联外部 launch 的源 commit；未设置时标记 `unknown`，不从目录名猜测 commit。Git 工作树仍保存真实 HEAD 和 tracked diff。快照的 `code_diff_available=false`，空 `CODE_DIFF.patch` 不表示已验证工作树无修改；文件缺失/读取失败和真实工作树中的 Git 错误仍会抛出。

只查看完整任务与配置（不训练、不创建结果目录）：

```bash
python -m experiments.osram_history_training_controls_20261014.run --dry-run --dataset IEMOCAPFour --seeds 66 67 68
python -m experiments.osram_history_training_controls_20261014.run --dry-run --dataset IEMOCAPSix --seeds 66
python -m experiments.osram_history_training_controls_20261014.run --dry-run --dataset MELD --seeds 66
```

后续获准执行时，单 fold 或专用 dispatcher：

```bash
python -m experiments.osram_history_training_controls_20261014.run --dispatch --dataset IEMOCAPFour --seeds 66 67 68 --gpu 6 --concurrency 1
python -m experiments.osram_history_training_controls_20261014.run --run-fold --dataset IEMOCAPSix --control causal-gru --seed 66 --fold 1
python -m experiments.osram_history_training_controls_20261014.run --dispatch --dataset IEMOCAPSix --seeds 66 --gpu 6 --concurrency 1
python -m experiments.osram_history_training_controls_20261014.run --dispatch --dataset MELD --seeds 66 --gpu 6 --concurrency 1
```

服务器归属沿用现有实验记录，不自动迁移；`--gpu` 指宿主机卡号，项目专用 dispatcher 始终拒绝物理GPU4，不依赖 hostname 包含 biggpu（该 SSH alias 实际 hostname 为 user22）。直接 `--run-fold` 在 user22/biggpu 或显式 `OSRAM_P2_SERVER_PROFILE=biggpu` 时要求设置非空 `CUDA_VISIBLE_DEVICES`，不能默认看全卡。

两个入口通过 `nvidia-smi` 的宿主机 index/UUID 清单构造健康卡白名单，排除物理index4和已知坏卡 `GPU-46fb379f-cc90-dc82-9b5e-5d011f552264`；坏卡完整UUID及前缀均拒绝。数字宿主机卡号、完整UUID和唯一UUID前缀都必须解析到唯一白名单卡，未知/歧义/重复选择拒绝。直接入口在导入 torch 前把解析结果规范为完整UUID，避免 `CUDA_DEVICE_ORDER` 影响数字映射；dispatcher 子进程显式 physical6 可正常使用，其进程内 `cuda:0` 从不用于识别宿主机卡号。provenance/status 保存解析后的宿主机 index/UUID。

dispatcher 保留 nvidia-smi GPU identity，默认并发1；CPU测试不构成GPU显存容量证据，正式执行前应按该服务器实际空闲资源与训练峰值调整并发。用 `--control local-only` 或 `--control causal-gru` 可只调度一组。重复输出、日志或 dispatcher status 会拒绝覆盖。

路径可通过 `OSRAM_P2_OUTPUT`、`ECERC_GCNET_FEATURE_ROOT`、`GCNET_DATASET_ROOT`、`GCNET_MELD_PICKLE` 指定；每个输出位于 `{dataset}/{control}/seed_{seed}/fold_{fold}`，不同任务隔离。
