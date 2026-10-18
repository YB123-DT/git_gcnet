# MMGCN MELD 特征核验与原 Flat 单种子实验

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC

完成更新：seed66 已训练 50/50 epoch。八率 ACC57.826/UA30.562/W-F1 53.540；
完整结果与核验见 `RESULT.md`、`FINAL_SUMMARY.json`。下文启动说明保留作复现记录。

用户要求：下载 ACL-IJCNLP 2021 MMGCN 发布的三模态 MELD 特征，与 ECERC
比较；不同则运行一次 MELD。本轮仅 seed66，不调参、不加模块、不扩大种子。

## 来源与比较结果

- 论文：<https://aclanthology.org/2021.acl-long.440/>。
- 作者仓库：<https://github.com/hujingwen6666/MMGCN>；研究组入口
  <https://github.com/AIM3-RUC/MMGCN> 指向该源码。
- 固定 commit：`85732d984f70c1c84dd47c81aa97f1271397b899`。
- 下载文件：`MELD_features/MELD_features_raw1.pkl`，88,193,675 bytes。
- 原文件 SHA256：`d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0`。
- 官方 Git blob SHA1：`55873a76e17fa3e054bcc5afc534cd1dde2eb2d8`；下载内容通过校验。
- 当前 ECERC SHA256：`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`。

| 内容 | MMGCN | ECERC | 全部 13,708 句话的比较 |
|---|---:|---:|---|
| Audio | 300维 | 300维 | 原始数组及 float32 训练数组均逐元素一致 |
| Visual | 342维 | 342维 | 原始数组及 float32 训练数组均逐元素一致 |
| Text | 600维 | 1024维 | 不同；维度就不相同 |
| Dialogue/话语顺序、标签、speaker、文本句子 | — | — | 全部一致 |

视觉全零 train/dev/test 分别为 3407/95/258，与 ECERC 完全相同。因此该来源
**不会修复视觉全零问题**。实验实际更换的是 MMGCN 的 600 维文本特征，
不能把潜在效果解释为音视频质量变化。

MMGCN 的旧十字段文件将 canonical train+dev 合并为一个 training-ID 集合，
而 test 与当前相同。这里只借用预提取特征，不照搬这个 training-ID 集合：
保持官方 train9989/dev1109/test2610、全部 1432 段对话，绝不将 dev 并入训练。
没有丢弃样本、伪造视觉、重新提取或额外归一化。预提取特征的上游训练细节
没有通过该 pickle 本身证明，不宣称它们必然是未经任务训练的通用特征。

## 转换和审计

`prepare.py` 校验固定下载 hash/Git blob、全部身份与元数据、原始 split 覆盖、
逐元素特征、有限值和全零分布。外部 pickle 只允许 NumPy 数组/dtype 和 set
等必要全局对象；受接受的 canonical 文件另行固定 SHA256 后读取。

输出只替换 canonical 17字段格式的特征字段；原 labels、speaker、句子和
split 直接保留，顺序不改。实际训练仍只消费现有 eroberta1 位置的 Text，
不新增输入流。转换输出 SHA256：
`d98f81538c1a2932ec3b959df6d02cf54930cf61063353174d284c9c1b8facd0`。

原始文件和转换数据在本地 `/data2/yb/paper/feature_sources/mmgcn_20261018/`，
不提交到 Git。完整审计记录见 `PREPARED.json`。

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m experiments.osram_meld_mmgcn_features_20261018.prepare \
  --source /data2/yb/paper/feature_sources/mmgcn_20261018/MELD_features_raw1.pkl \
  --canonical /data2/yb/paper/baseline_sources_20261006/ECERC/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output /data2/yb/paper/feature_sources/mmgcn_20261018/NEW_mmgcn_canonical_meld.pkl
```

转换是本地 CPU-only NumPy1.26.4；训练仍使用 biggpu 原环境，不能混为环境迁移。
文件已存在时拒绝覆盖；如果三种 float32 输入完全相同，则只保存审计，不生成
重复训练数据。当前比较明确不同，因此执行获授权的一组训练。

## 固定训练协议

- biggpu；`/data2/yb/conda_envs/cmerc-repro/bin/python`；Torch1.13.1+cu117、NumPy1.26.4。
- 复用已验证的特征切换 worker，不修改模型/训练器：
  `experiments/osram_meld_alternate_features_20261017/worker.py`，SHA256
  `66fc911607d06ebc98b7e7ab407aaf786185586eebcd8cb2ac4ab4e6b46a48d1`。
- worker 的不可变数值源码是 `737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`，
  运行前验证其 131 个源码/配置文件 hash。与此前 MAGTKD 数据切换实验一致，
  不是暗中改变训练器；本轮当前 Git SHA 另行写入启动记录。
- `CONFIG.json` 与上述原 Flat 配置完全一致：seed66、50 epoch、batch32、
  Adam constant LR=.001、weight decay1e-5、dropout=.5，不同时引入 cosine。
- `gcnet-random-mask-v1`，host/guest 原始采样、cyclic0.0–0.7；不使用 fixed mask。
- 原 Flat1600、Local Skip、8 heads、K/V64、write_step=.6、原联合 Block Delta
  Write、causal read-before-write。emotion-only/no-JEPA，没有 Block2/Nested/Gate。
- Text 输入维度随真实特征从 1024 变 600；对应 Projector 输入参数量会改变，
  不是人为再调模型宽度。不同输入维度也会改变初始化 RNG 的消耗，因此不能
  声称它与 ECERC 是逐参数初始化及测试 mask 都完全相同的配对运行。
- 逐率 Test-best ACC；UA 是同一选中 epoch/同一 mask 的七类 Macro Recall。
  保存原 BEST、exact selected predictions/masks 及完整恢复状态，最终从这些
  保存的预测计算指标，不额外随机重测来补 UA。

资源：健康物理 GPU5，UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`，
进程内 cuda:0；GPU4 禁止。本轮一组训练，先做复用 worker 自带的三 batch
真实梯度检查。不同进程独立初始化，不将 smoke 状态带进正式训练。
不停止其他任务、不修改环境、不删除受保护权重。

## 远程入口与状态

新输出：`/home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018/`。
`run.sh` 通过独占 pipeline 锁运行 smoke → 正式单种子训练；失败即停止，
不改超参数偷偷重试。复现前将已有 worker、CONFIG 和转换数据放到注明目录。

```bash
bash experiments/osram_meld_mmgcn_features_20261018/run.sh
```

真实启动进度以远程 SUMMARY/history/进程为准；本文说明入口，不代表训练已完成。
最终 `RESULT.md` 应包含八率 ACC/UA/W-F1、high mean、真实环境与局限。

## 最小测试

新转换测试保护 metadata/dev 不变、拒绝错标签/缺话语/错误 split、相同输入不重复训练：

```bash
python -m unittest discover -s tests -p test_meld_mmgcn_features.py
```

复用 worker 的测试还检查环境绑定、源码不可变、CUDA RNG 修复和同次选点指标。
本机默认 Python 未安装 sklearn，worker 指标测试改用已有的 cmerc-xp-repro
执行，不安装依赖或修改任何环境。正式训练不使用这个本地环境。
