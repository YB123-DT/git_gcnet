# 交付代码审计范围

本分支从目标仓库main隔离建立，新增代码位于experiments/baseline_5models_local_20261015，根.gitignore仅追加该实验输出目录排除。没有合并其他实验分支、切换用户工作分支、修改正在训练的源码快照或上传数据/权重/日志/结果。

## 来源与算法边界

- MMIN：adapters/mmin/provenance.json与audit/mmin_audit.md。官方master 69a45fed与既有GCNet话语特征适配对照；保留Teacher、ResidualAE、互补缺失目标与Cycle。
- MoMKE：adapters/momke的UPSTREAM_COMMIT与audit/momke_audit.md；官方2dceb57。Original等价参考附两个原文件与SHA，Causal只改变Attention可见性。
- ComP：adapters/comp/SOURCE.json与audit/comp_wrapper.md，官方28192d3，既有因果实现aa53831。
- GCNet/SDR：adapters/graph_models/AUDIT.md、各provenance.json和adaptation.patch。GCNet官方8cce4a0与本地原四文件逐字节一致；SDR b455461必须标Implementation-fixed。

本地历史实验的详尽清单、真实运行状态和产物仅保留本机，未包含在交付包中。不能从本分支代码或测试通过推断正式训练已经完成。

## 发布前独立检查

发现并修复续训数据身份校验、Teacher阶段快照沿用、Teacher缓存签名、图模型配置别名四项边界问题；每项有回归测试。没有更改冻结参数、模型数学、数据划分、Mask采样规则或Test-best协议。具体位置见REVIEW_GUIDE.md。

测试与验证命令写在README；最终提交的Tested trailer记录本次实际结果。测试产生的临时数据与日志不提交。
