# 已确定 Nested：四数据集冻结诊断

INTERNAL DIAGNOSTIC ONLY。沿用现有 per-rate BEST Test-oracle，不更改选模或模型。

范围：MOSI/MOSEI seeds66/67/68，IEMOCAP4/6 seeds66/67/68 ×五 session folds，共36个源run。复用 `nested_gnn_rooted_evidence`，不是 root-aware 或 Local8；不训练、不含 Flat 对照。

六项：七种固定可见组合；六种整段缺失的 Gap 开关；六种整段缺失的 Base 开关；两条 Local 直接分类路径同时关闭；Query/Addressing/Read 余弦；同 checkpoint 的 Local-only/Local+Base/Full 三读出。随机八率与固定组合分别报告，不混合两种条件。

## 干预定义

一次正常 encoder/causal OSRAM scan，缓存原始 Local/Base/Gap，后续只重算 Nested 与分类读出。关闭 Base/Gap 时先移除 Nested 输入中对应证据和 active nodes，再在输出 Adapter 对应槽位安全清零，不能以“原读值已置零”为借口允许 decoder bias 或 embedding 重新生成该槽。保留原 availability，绝不改变 Memory/query/write。

`local_only`：屏蔽全部历史，Nested bypass，不从 type embedding 构造假历史；保留原 Local Adapter 与完整 Local Skip。`local_base`等于`gap_off`。`no_local`：Nested 输入 Local 置零，最终 Adapter Local 置零，完整 Skip 输出置零；当前输入仍影响上游 query/写入，所以不是纯历史模型。

这是严格移除某路分类证据的干预，比旧 Flat 的仅输出槽清零涉及额外的 Nested 跨节点路径；结果描述为本轮定义下的条件贡献，不称训练因果效应。

固定 A/T/V 用原 `.7` BEST，AT/AV/TV 用 `.3`，ATV 用 `.0`；这些是 checkpoint 标签，不是在固定组合测试中再次随机删除的比例。当前模态特征不补全。padding/inactive Gap 必须安全置零。

Query audit 读取真实 `_scan()` 原 query、residual addressing diagnostics 及 Nested **之前**的 read。raw qB/qG 在 addressing 之前，不能用 raw-query/read 余弦差声称 Memory 映射坍塌。未定义零向量余弦保留 missing count，不补0。逐率、target、head 汇总；跨率记录不是独立样本。

## 结果口径

MOSI/MOSEI 按源任务非零标签与 prediction>0 计算 W-F1/ACC；IEMOCAP使用 argmax 的 W-F1/ACC/UA。IEMOCAP先等权平均5folds，再rates，再3seeds；SD为seed均值样本标准差。不把fold或rate当独立seed。所有原预测、权重保留且不覆盖。

## 启动及产物

所属服务器biggpu；物理GPU6白名单，最多2组并发，MOSEI准入18000MiB，其余6500MiB，不改batch。GPU4禁止。source目录在首次正式启动后不修改。

```bash
cd /data2/yb/remote_experiments/osram_nested_diagnostics_20261007/source
/data2/yb/reproduction_workspace/envs/s0/bin/python -m experiments.osram_nested_diagnostics_20261007.dispatch --verify-sources
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m experiments.osram_nested_diagnostics_20261007.dispatch
```

`SOURCES.json`：36源run配置、metrics、288checkpoint、288原预测SHA256。
`DISPATCH.json`：实际PID、GPU/UUID、命令、日志、planned/running/complete/failed。
`results/*/STATUS.json`、`SUMMARY.json`、NPZ和query CSV：单run验证和结果。
`DIAGNOSTIC_SUMMARY.json`、`RESULT.md`：自动更新，未齐全时不生成三种子完整均值。

本目录的代码是独立评测，不修改生产训练代码。正常重放、mask/labels、source predictions、模型权重不变、inactive/padding安全是提交运行前后的必要检查，测试通过不等于性能有效。
