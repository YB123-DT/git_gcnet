# MELD 原始视觉全零审计

INTERNAL DIAGNOSTIC ONLY

## 范围与证据

本次只读取受信任的 ECERC 原始 pickle 和已有 W0 的逐率 ACC-selected
预测。没有训练、加载模型推理、采样新 mask、占用 GPU、修改原始特征或正在运行的实验。

- 审计时分支：`feature/ecerc-features`；起始 HEAD：`619326e9932393fcf00dd7fa1c7307082fa62a08`。
- pickle：`/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl`。
- SHA256：`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。
- 原始视觉：342 维、float64；全零定义为整条原始向量精确等于 0，排除 padding。
- 全部 A/T/V 原始值有限；float32 转换未改变任何视觉全零标记。
- 完整统计与输入 hashes：`AUDIT.json`；实现：`audit.py`。

## 1. 不只是某个情绪类别的问题

| Split | 话语数 | V 全零数 | 比例 | 含全零话语的对话 | 整段 V 全零对话 |
|---|---:|---:|---:|---:|---:|
| train | 9989 | 3407 | 34.108% | 848/1038 | 34 |
| dev | 1109 | 95 | 8.566% | 50/114 | 0 |
| test | 2610 | 258 | 9.885% | 136/280 | 1 |

训练集七类内部全零比例为 29.85%–34.99%，复核了用户提供的计数：

| 情绪 | 总数 | V 全零数 | 类内比例 |
|---|---:|---:|---:|
| Neutral | 4710 | 1648 | 34.99% |
| Surprise | 1205 | 411 | 34.11% |
| Fear | 268 | 80 | 29.85% |
| Sadness | 683 | 220 | 32.21% |
| Joy | 1743 | 584 | 33.51% |
| Disgust | 271 | 92 | 33.95% |
| Anger | 1109 | 372 | 33.54% |

更明显的是 train/dev/test 之间的比例差异。训练集按全局对话 ID 分箱：

| ID 区间 | 话语数 | V 全零数 | 比例 |
|---|---:|---:|---:|
| 0–99 | 949 | 155 | 16.33% |
| 100–199 | 946 | 155 | 16.38% |
| 200–299 | 931 | 238 | 25.56% |
| 300–399 | 889 | 338 | 38.02% |
| 400–499 | 932 | 370 | 39.70% |
| 500–599 | 994 | 395 | 39.74% |
| 600–699 | 1058 | 441 | 41.68% |
| 700–799 | 844 | 321 | 38.03% |
| 800–899 | 988 | 425 | 43.02% |
| 900–999 | 1032 | 401 | 38.86% |
| 1000–1038 | 426 | 168 | 39.44% |

ID 分箱只表示文件中的序列位置，不能直接等同于采集时间、电视剧集或提取批次。
训练集对话内最长连续 V 全零为 9 句，test 为 6 句；统计不跨对话。
完整对话、位置、speaker 槽位分布均在 JSON 中。
**speaker one-hot 的槽位不能直接当作跨对话的固定人物身份。**

Audio 全零仅训练集 1 条；Text 三个 split 均未发现全零。

## 2. 与 GCNet 随机缺失必须区分

原始 V 全零独立于 `gcnet-random-mask-v1`。当前实现没有自动把全零 V
改为缺失：只要 availability 中 V=1，它仍被视作可见。

代码依据：

- `gcnet_modality_jepa/dataloader_meld.py`：读取原始特征，转换 float32；
  同一特征复制到 Host/Guest，不按特征是否为零修改 availability。
- `gcnet_missing_m3/model.py::ObservedSetEncoder.forward`：以有效 utterance 和
  availability 选择模态，没有原始全零检查。
- `ModalityProjector` 含 LayerNorm、带 bias 的 Linear 和非线性；因此不能把
  「原始 V 全零」直接理解成「任务侧视觉 latent 为零」。
- `gcnet_missing_m3/osram.py::_project_tokens`：Key 同时依赖该模态 latent、
  当前 write node、availability 和 speaker；Value 依赖模态 latent。
  **不能因为原始 V 相同就断言其写入 Key 也相同。**

全零 V 可能经过学习变成默认表征并进入真实观测写入路径；这说明数据质量与
mask 语义存在需要核查的区别，不足以证明其写入有害。
本次未把全零 V 改为缺失，也未修改任何联合 Block Write。

## 3. 已有原 Flat 预测分层：不是 Memory 增益表

使用 W0 seed66、50 epoch、保留 Local Skip 的 Flat1600，GCNet random mask。
数值源码固定为 `b4a313841f098487a6482d2712bdac94aa8e319e`。
输入目录：
`/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0`。
这是本轮 W0，不替换用户已接受的三种子 60.693 baseline。

每率使用原来选中 ACC 最佳 epoch 的同批 logits、labels、availability；
没有额外随机测试。按 `(conversation_id, utterance_index)` 连接原始全零标记，
严格核对完整 test 集身份、标签、speaker、文件 hash、mask hash 和选中 epoch。
重算完整 test 的 ACC/UA/W-F1 与已有 SUMMARY 一致。

以下是 **Full ACC (%)**，括号为样本数：

| rate | 原始 V 全零且 V 可见 | 原始 V 非零且 V 可见 | 随机 mask 标记 V 缺失 |
|---|---:|---:|---:|
| .0 | 76.357 (258) | 65.434 (2352) | — (0) |
| .1 | 72.609 (230) | 64.120 (2107) | 64.469 (273) |
| .2 | 74.874 (199) | 62.353 (1870) | 63.031 (541) |
| .3 | 67.539 (191) | 60.235 (1617) | 63.092 (802) |
| .4 | 66.667 (147) | 56.787 (1444) | 63.690 (1019) |
| .5 | 65.289 (121) | 56.148 (1163) | 59.955 (1326) |
| .6 | 54.808 (104) | 51.282 (897) | 59.291 (1609) |
| .7 | 50.725 (69) | 46.494 (770) | 59.289 (1771) |

对应固定七类 UA (%)，没有支持的类按 0 处理；不能忽略下文的支持数差异：

| rate | V 全零且可见 | V 非零且可见 | V 缺失 |
|---|---:|---:|---:|
| .0 | 49.727 | 47.478 | — |
| .1 | 42.233 | 41.359 | 43.525 |
| .2 | 45.423 | 37.955 | 38.238 |
| .3 | 33.765 | 35.595 | 38.127 |
| .4 | 29.541 | 29.457 | 38.843 |
| .5 | 33.804 | 26.921 | 31.823 |
| .6 | 16.952 | 18.895 | 33.645 |
| .7 | 16.327 | 14.286 | 37.188 |

没有观察到「V 全零组整体 ACC 必然更低」。**这不表示全零视觉有益**：
不同组的类别、A/T 是否可见、样本难度不同，不能从跨组绝对 ACC 推出视觉或
Memory 的因果效用。特别是 .7 中 V 可见等于 V-only，而 V 缺失组包含 A-only/T-only。

JSON 同时保留 CE、W-F1、七类 support/recall、各组 availability 组合计数。
测试集 V 全零样本没有 Disgust，Fear 仅 3 条：子组 UA 必须同时注明支持数。
本报告同时保存固定七类、无支持类补零的 `UA_7_zero_filled`，以及只平均有支持
类别的 `UA_supported`，不把后者混充完整七类 UA。

### 尚缺什么

本次可严格对齐的 W0 选点文件只有 Full 预测，不能据此计算同 checkpoint 的
Local → Local+Base → Full 纠错／致错及历史增益。
历史七读出文件使用旧 legacy-fixed 协议，且缺少完整身份和 availability，
不能凭标签顺序相同就混进这张当前 random-mask 表。

因此，**split/类别/ID/输入槽位的原始数据统计和 Full 错误分层已完成；真实
人物分布、原视频抽查以及当前协议的三路径历史增益分层尚未完成**。
若补齐三路径分层，只需复用 W0 原 checkpoint 和 exact selected mask 做冻结
三读出重放，不需重训；本次没有启动该推理。

## 4. 原视频与结论边界

本地官方 CSV 与 pickle 的映射、人物身份和原视频库存另见 `VIDEO_INVENTORY.md`。
没有视频画面证据时，不能将全零归因为无脸、说话人不可见、视频损坏或提取代码失败。

目前能确定的是：原始 V 全零广泛存在、train 比 dev/test 明显更多、训练 ID
区间分布不均，且 raw-zero 不等于 sampled-missing。
**尚不能确定：全零来源，以及它是否解释 MELD Memory 的弱贡献。**

## 复现与验证

在 biggpu 的原环境运行，CPU-only：

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python audit.py \
  --pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --predictions /data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0 \
  --output /tmp/NEW_AUDIT.json
```

输出路径必须不存在；不会覆盖旧结果。此次实际远程输出：
`/tmp/meld_visual_audit_20261017.igPAyI/AUDIT.json`，已回收进本目录。

测试：`python -m unittest discover -s tests -p test_meld_visual_zero_audit.py`。
验证身份对齐与拒绝错误标签、三组严格分区、缺类 UA 的显式口径。
另以当前远程环境的 sklearn 独立重算八率完整 test 的 ACC、七类 UA、
W-F1 和 CE，8/8 与审计输出一致。
不做统计显著性或机制效果声明；没有新增模型对照。
