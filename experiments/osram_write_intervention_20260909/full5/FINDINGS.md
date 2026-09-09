# IEMOCAP-4：四组 frozen-checkpoint intervention

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

## 结论

本轮不支持“DynamicGlobal 稳定优于 Fixed0.9”。Fixed0.9 在五个已测
rate 的平均结果中均最好，相对 Reference 的五率均值提升在 5/5 seeds
成立；DynamicGlobal 相对 Fixed0.9 仅 1/5 seeds 为正。

因此暂不引入 adaptive write strength，也不训练 protected-memory 新模型。
下一研究重点应是原 ridge write 的有效步长，而不是先包装复杂保护机制。
这并不证明 0.9 最优，也不证明所有动态策略都无效。

## 协议

- IEMOCAPFour，fold 5，seeds 66–70。
- 沿用首轮五个 rate：0.0 / 0.1 / 0.3 / 0.5 / 0.7，**不是八率均值**。
- 每个 seed 使用已有的一个 checkpoint，epoch 分别为 33/54/38/30/64。
- 共 100 次组合评估；没有训练、重新选择 checkpoint 或调参。
- checkpoint 原先按 Test-oracle 选择，且 IEMOCAP 的 held-out Session
  同时作 validation/test，因此只能作为内部诊断。
- Fixed0.9 对所有 write 乘 0.9，包括 complete/no-history；其余两个干预
  在没有历史缺失地址时保持原写入。此差别是锁定对照定义，不是结果后修改。

## W-F1（%）

| miss | Reference | Fixed0.9 | DynamicGlobal | Protected |
|---|---:|---:|---:|---:|
| 0.0 | 84.3344 | 84.4714 | 84.3344 | 84.3344 |
| 0.1 | 83.9241 | 84.2094 | 83.9552 | 83.9243 |
| 0.3 | 81.9933 | 82.1925 | 82.0897 | 81.9091 |
| 0.5 | 80.1181 | 80.3385 | 80.1910 | 80.3108 |
| 0.7 | 77.7502 | 78.1223 | 78.0661 | 78.0364 |
| 五率均值 | 81.6240 | **81.8668** | 81.7273 | 81.7030 |
| seed 均值的样本 SD | 0.5384 | 0.5197 | 0.6538 | 0.6140 |

先在每个 seed 内对五个 rate 等权平均，再统计五个 seed；不把 25 个
seed/rate 组合或数百万 head records 当成独立重复。

| seed | Reference | Fixed0.9 | DynamicGlobal | Protected | Dynamic−Fixed（百分点） |
|---|---:|---:|---:|---:|---:|
| 66 | 81.7843 | 82.0108 | 81.9131 | 81.9028 | −0.0977 |
| 67 | 81.4469 | 81.5460 | 81.5827 | 81.5001 | +0.0367 |
| 68 | 81.6469 | 81.8342 | 81.6780 | 81.7090 | −0.1562 |
| 69 | 80.8788 | 81.2883 | 80.8224 | 80.8570 | −0.4659 |
| 70 | 82.3632 | 82.6548 | 82.6401 | 82.5461 | −0.0147 |

Fixed−Reference 平均 +0.2428 个百分点。Dynamic−Fixed 平均
−0.1396 个百分点，配对差值 SD 0.1970；描述性双侧符号翻转检验 p=0.1875。
N=5 的双侧检验分辨率很低，不据此宣称等价或显著性，也不把这一级别的
提升说成大型突破。完整逐 seed/rate 数据见 `per_seed_rate.csv`。

## 当前 observed write-fit 与历史 retention

当前 observed slot 的相对误差为 `||M k − v|| / (||v|| + eps)`。
`fit_gain = err_before − err_after` 为正表示新写入改善当前拟合。
`err_original_after` 是同一个当前状态下、不施加干预的假想 write 结果，
不是从 Reference rollout 借来的状态。

下表为 miss=0.7，各 run 内按有效 slot/head 记录平均，再对五 seeds 等权：

| 模式 | 历史 probe 实际读取误差 | 历史 write damage | 当前 observed 写后误差 | 当前 observed fit gain |
|---|---:|---:|---:|---:|
| Reference | 0.207436 | 0.140678 | 0.001865 | 0.878512 |
| Fixed0.9 | 0.244474 | 0.107762 | 0.086785 | 0.765094 |
| DynamicGlobal | 0.224230 | 0.108802 | 0.064065 | 0.796149 |
| Protected | 0.145065 | 0.000026 | 0.115974 | 0.727767 |

1. Protected 确实保住历史地址，但牺牲较多当前 observed write-fit，且
   没有获得最好的任务分数。地址保护与新信息写入存在可观测的取舍。
2. Reference 当前写后误差几乎为零，但 F1 并非最好；极佳的当前
   association 拟合不是最优情绪表示的充分条件。
3. Fixed0.9 的历史单步 write damage 较小，但实际读取时历史误差反而
   高于 Reference。因此不能只挑 damage 下降就声称总体 memory 更好。
4. 这些是固定权重下的干预结果，仍有分布偏移；不证明任务提升的唯一
   因果机制，更不能直接推断重新训练后的收益。

完整 rate/模态/seed 的四种 write-fit 指标、retention、更新范数与配对
差值见 `summary.csv` 和 `RESULT_FULL5.md`。写后误差仅可影响后续读取，
不与当前 utterance 的分类误差建立直接因果绑定。

## 已有 checkpoint 的步长静态检查

直接读取五个 checkpoint 的 learned beta，未修改参数：

| seed | beta 范围 | 单 observed normalized-key 有效系数范围 |
|---|---|---|
| 66 | 0.482058–0.497182 | 0.997930–0.997993 |
| 67 | 0.470806–0.496248 | 0.997881–0.997989 |
| 68 | 0.477099–0.498729 | 0.997908–0.997999 |
| 69 | 0.483131–0.497076 | 0.997934–0.997992 |
| 70 | 0.468768–0.497329 | 0.997871–0.997993 |

五个 checkpoint 的 write ridge 都是 0.001。单 normalized key 时，
修正系数为 `beta / (beta + lambda_w)`，约 0.998；不是 beta≈0.5
就只更新一半。Fixed0.9 会将该单-key 系数降至约 0.898。
多 observed slots 的 block solve 有 key 相关性，不能把这个单-key
数值当作所有方向统一的真实步长。这项检查给出后续研究依据，不宣称
已证明训练存在 bug；本轮没有修改 beta、ridge 或训练方式。

## 验证、代码与数据

- 202 个相关模型/诊断测试通过；4 个纯统计测试通过。
- 五个模型均验证全 state-dict tensor 不变、同 seed/rate 四组 mask 相同。
- miss=0 的 Reference/DynamicGlobal/Protected logits 逐元素一致；
  Fixed0.9 不在此等价要求内。
- seed66 旧三组的 15 份 task metric 字典和 mask hash 与首轮完全一致。
- 校验 300 份 gzip JSONL、3,935,920 条记录；数量与 metadata 一致，
  current write-fit 只包含 observed slots，fit_gain 恒等式成立。
- 最大同状态 protected/global 更新范数匹配误差 3.8147e-6（FP32）；
  不声称不同 rollout 后续状态的更新范数仍相同。
- `gcnet_missing_m3/write_intervention.py`：仅评估的 Fixed0.9 与 write-fit。
- `gcnet_missing_m3/evaluate_write_intervention.py`：四组配对评估和记录。
- `../summarize_five_seed.py`：无模型依赖的统计汇总。

GitHub 保留代码、所有 metric summaries、checkpoint/mask provenance、
逐 seed/rate CSV 和 `RAW_MANIFEST.csv`。新增约 234 MiB 的逐 head gzip
原始记录不重复塞入 Git 历史；完整文件同时保存在：

- 本地：当前目录各 `iemocap4_seed*/` 子目录；
- biggpu：`/data2/yb/remote_experiments/osram_write_intervention_20260909/full5/`。

manifest 给出全部原始文件的相对路径、字节数和 SHA256。该保留策略
不删除任何原始记录；旧首轮已提交文件保持原状。
