# MOSI Frozen-Feature Downstream Plateau Summary

## 结论

在当前 CMU-MOSI、frozen wav2vec/DeBERTa/MANet、单模型、统一 checkpoint、八个
missing rates 同批训练、validation-only 选模的锁定边界内，本轮没有找到可稳定替代
deterministic Legacy control 的 downstream 改动。该证据现已同时覆盖局部模块与一次
等活跃参数的 whole-backbone replacement；结论仍不外推到其他数据集、上游特征适配或
不同 benchmark protocol。保留的可复现配置为：

```text
shared readout
target-balanced JEPA
legacy recurrent semantics
MSE task loss
independent Temporal/Speaker postgraph BiLSTMs
uniform JEPA rate coefficients
raw GraphConv messages
deterministic relation IDs and PYTHONHASHSEED=0
```

其五种子 validation 八-rate mean W-F1 为 `78.7675`，high-missing 为 `74.9589`，
miss-0 为 `85.6461`。这是 reproducible control，不是新的方法贡献。

## 锁定筛选总账

Option A--C 使用早期 historical controls，其 manifest 没有记录锁定后的 relation-ID
provenance，因此只作为方向性证据。严格平台判断主要依据与 direct deterministic
controls 配对的 D--H，尤其是完成五种子确认的 Option F。

| Option | 唯一变量 | 结果 | Validation delta (point) | 关键失败项 |
|---|---|---|---:|---|
| A | availability low-rank readout | EXPLORATORY FAIL* | -0.3655 | overall、seed stability |
| B | availability affine hidden residual | EXPLORATORY FAIL* | -0.6261 | 0/3 positive |
| C | utterance-balanced JEPA regression | EXPLORATORY FAIL* | -0.4190 | overall、high-missing |
| D | packed recurrent semantics | FAIL | +0.0907 | worst seed -1.6298 |
| E | SmoothL1 task loss | FAIL | +0.2464 | 1/3 positive、high-missing |
| F | shared postgraph BiLSTM | SCREEN PASS → 5-seed FAIL | -0.1785 (5 seeds) | 2/5 positive、high-missing |
| G | sparsity-weighted JEPA coefficients | FAIL | -0.2526 | 1/3 positive、high-missing |
| H | branch graph-message calibration | FAIL | +0.0620 | below +0.40、high-missing |
| I | equal-active full-context SDT backbone | FAIL | -1.2100 (5 seeds) | 1/5 positive、全部八率均值下降 |

Option F 是唯一三种子通过者，但新增 seeds 69/70 后从 `+0.5344` 反转为
`-0.1785` point，证明小 seed screen 不能替代五种子稳定性确认。

Option I 将 GCNet conversation backbone 整体替换为参数差仅 `+0.07963%` 的
full-context Pre-LN Transformer，同时保持 Observed-Set、Missing-M3、EMA、loss、mask
和优化协议不变。其五种子 validation 8-rate mean 为 `77.5575`，相对 control 下降
`1.2100` point；high-missing 下降 `1.4334` point，miss-0 下降 `0.7392` point，仅
seed 68 为正。完整逐 seed/逐 rate 证据见
`gcnet_missing_m3_sdt_backbone/results/SUMMARY.md`。

## 强平台证据的含义

该结论只适用于当前 CMU-MOSI frozen-feature downstream 边界，不表示 MOSI、
上游特征适配或不同协议不可能提升。严格配对证据覆盖 recurrent
padding semantics、robust task loss、跨图 sequence sharing、JEPA rate 聚合与
graph-message calibration，并由 Option I 扩展到一次整体 conversation backbone 替换；
A--C 对 readout、hidden affine 和 JEPA target 聚合的结果仅提供方向性补充。继续扫描
同族小变体或第三个对话主干会增加选择偏差，缺少新的机制依据。

此前 CaM-HG 的数值不能作为同协议硬门槛：其 feature dimensions/Audio extractor、
二分类目标、连续混合率采样、text blindness、test-time completion 与 seed 报告方式均
与本协议不同。若要继续追求那组数值，必须新建 upstream/protocol reproduction lane；
不能在本 lane 中用 test 选模型或继续堆 downstream trick。

## 数据完整性

- A--H 均按八-rate validation mean 选择最早最佳 epoch；
- failed candidates 未因结果不佳而解封 test；
- 新候选均使用 direct deterministic controls，且非 treatment 配置审计通过；
- F 的五种子 config/history SHA256 已固化；
- G/H 均完成 100 epochs，无非有限值、常量输出或单 sign 坍塌；
- G/H 的 artifact SHA、source provenance、MOSI 单 speaker relation 映射校正及 H 的
  最佳 checkpoint 参数诊断记录在 `PROVENANCE_AUDIT.json`；
- 本地结果不包含大 checkpoint，remote checkpoint 保留用于可追溯审计。
