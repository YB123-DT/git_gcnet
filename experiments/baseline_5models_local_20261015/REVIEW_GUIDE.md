# GPT代码审查指南

请审查此分支相对main的差异，重点仅为experiments/baseline_5models_local_20261015；根.gitignore只用于排除产物。此包不含正式实验成绩，不能据此评判性能。

## 推荐阅读顺序

1. PROTOCOL.md、configs/frozen_defaults.json、configure.py：七版本、三数据集、三Seed、八MR、五Session划分、两阶段预算与测试oracle边界。
2. common_data.py：A/T/V特征字段、标签/说话人、固定Mask的身份键、实际MR。
3. adapters/momke + tests/test_momke_causality.py：未来特征/未来Mask扰动、Prefix/Padding/Batch、Stage1/2、数值与梯度。
4. adapters/comp：是否忠实复用已有因果分支，Original标记与Prompt，stage_loss是否保持原始训练行为。
5. adapters/mmin：Teacher互补缺失编码目标、ResidualAE/Cycle、禁止Teacher参与预测、跨MR身份。
6. adapters/graph_models：MELD九说话人关系、双图/超图路径、SDR修复命名与Attention轴、分类和重建损失轴。
7. train_one.py/runtime.py：同Epoch四指标、Stage2-only Test-best、早Epoch tie-break、不可变Checkpoint、完整恢复与预测回放。
8. dispatch.py/supervise.py：本机限制、UUID空闲判断、文件锁、PID归属、重启、失败退出与预算决策只读训练loss。
9. aggregate_results.py：五折预测先合并，三个Seed再ddof=1；不完整Seed不可冒充完整；协议/Mask/样本相同才允许配对。

## 交付阶段补齐的边界

- 当前checkout产物与缓存隔离，防止审查副本指向正在运行的输出目录。
- 恢复训练之前核对原数据内容和划分，拒绝静默混用数据版本。
- Teacher阶段即使尚未生成Missing Model的last.pt，也沿原不可变快照恢复。
- Teacher身份覆盖实际优化器和依赖实现，修改协议不能误复用；同身份仍跨MR共享。
- 统一drop_rate/recon_weight和官方dropout/recon_loss_weight兼容，冲突值必须报错。
- Original等价测试的官方参考随包提供，CPU导入不启动训练；实际训练入口仍强制本机/两UUID。

这些是交付分支的可审查修复，正在运行的本地实验继续原快照；没有将新分支悄悄替换进活跃训练。

## 必须保留的诚实标记

Test-best ACC属于test_oracle，存在测试集选择乐观偏差。SDR为Implementation-fixed；MELD GCNet为MELD-adapted Original；MMIN为UTT特征适配。MoMKE因果/原版仅时间可见性不同。小规模检查不代表性能提升或完整三种子结果。

## 已知上游行为

ComP跨Batch Prompt混合与未命中的梯度调制参数模式被保留，不能无依据宣称Batch独立；原两阶段历史回载也是no-op。两图模型继承未packed双向RNN的Padding长度敏感性。SDR超图默认分支有未用参数，未擅自删除。原始数据缓存的文件变更检测与严格内容身份是重要审查点。审查应区分这些公开的原方法限制与本次实现错误。

上游复制文件保留原始字节、CRLF与末尾空格，维持SOURCE/provenance中的SHA校验；本目录.gitattributes只对这些文件及补丁上下文豁免空白风格检查，作者新增的Runner/Wrapper/测试仍正常检查。
