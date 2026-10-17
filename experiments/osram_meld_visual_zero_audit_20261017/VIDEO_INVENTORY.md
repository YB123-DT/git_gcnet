# 原视频库存与身份映射边界

审计日期：2026-10-17。只读检查，不下载视频、不运行模型。

## 已找到的元数据

本地目录：`/data2/yb/paper/feature_sources/mmalign_20261017/`。

| 文件 | SHA256 |
|---|---|
| train_sent_emo.csv | d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db |
| dev_sent_emo.csv | 2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c |
| test_sent_emo.csv | 8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888 |

只读库存核查将这些 CSV 与本地同 SHA256 ECERC pickle 比较，确认全部
13,708 条话语的标签、长度和对话 ID 集合匹配；文本在现有规范化规则
（Windows-1252 引号、NFKC、空白）处理后逐句一致。
本地校验 pickle 位于
`/data2/yb/paper/baseline_sources_20261006/ECERC/data/meld/meld_emotion_semantic_features_roberta.pkl`。
其 SHA256 经实际计算，与远程被接受的文件一致。
映射使用各 split 的实际 CSV，不能在没有核对时套用数字偏移：

- train：pickle 全局 dialogue ID = CSV Dialogue_ID。
- dev：pickle 全局 dialogue ID = CSV Dialogue_ID + 1039。
- test：pickle 全局 dialogue ID = CSV Dialogue_ID + 1153。
- 话语身份使用匹配 CSV 行的真实 `Utterance_ID`，不要直接假设列表索引就是视频编号。

因此，查找候选视频时可以使用匹配行的 `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`。
**文件名必须带 split 才是完整身份**：train/test 都有 `dia0_utt2.mp4`，不能跨 split 合并。
这是元数据身份映射，不表示对应视频实际已找到、可解码或能检测到说话人。

## Speaker 槽位不能当作固定人物

官方 CSV 的人物名比较显示，pickle 的 speaker slot 0、1 分别对应多个不同人物。
因此 `AUDIT.json` 的 `by_speaker_slot` 只能解释输入槽位，不能称作固定人物统计。
本次未交付按真实人物姓名的全零比例表。

## 原始视频未完成抽查

限界检查如下，均未找到可用于抽查的 `dia*_utt*.mp4`：

- 本地 `/data2/yb/paper/feature_sources`，最大深度 5。
- 本地 `/data2/yb/paper/baseline_sources_20261006/ECERC/data/meld`，最大深度 5。
- biggpu `/data2/yb/paper/remote_runs_20261012/data/meld`，最大深度 3。
- biggpu `/data2/yb/paper/remote_runs_20261012/data`，最大深度 4。
- biggpu `/data2/yb/paper/CSS_causal_meld`，最大深度 4。
- biggpu `/data2/yb/paper/SDT_causal_meld`，最大深度 4。
- biggpu `/data2/yb/paper/reproduction_meld_causal`，最大深度 4。
- biggpu `/data2/yb/paper/reproduction_meld_noncausal`，最大深度 4。

这不是全服务器文件系统穷尽搜索，也不是声明数据集原视频不存在。
本地 CSV 所在的同名 `feature_sources` 路径在 biggpu 未找到。

目前没有画面证据判断：全零是否来自看不到说话人、无人脸、原视频异常、
人脸提取失败或其他预处理规则。**不能把任何一种来源写成已确认结论。**
