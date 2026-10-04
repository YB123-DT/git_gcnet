# OSRAM 新增40项实施计划

目标：交付40项来源可追溯、独立可切换的读出适配，而不是40项性能承诺。用户已授权直接实现，本次不含正式训练。

架构：沿用MeaningfulInputAdapter的Flat输入边界、HeadTokenizer、active_groups、safe_mask。保留原Local skip、Memory读写query、任务头/loss、mask与旧ID；新增方法默认关闭，最终修正映射零初始化。不新增依赖。

文件分工：源码卡保存在docs/osram_new40_*cards.json和nine_designs.json；registry列出40个ID；meaningful_input_new40.py作延迟分派与公共解码；各meaningful_new40_<family>.py只承载独立核心。meaningful_input.py只注册新族。实验目录提供CATALOG、参数量、报告、prepare选择工具，复用旧训练器。

步骤：

- 完成原文/代码与历史清单去重。允许同假设，不允许同核心改名。
- 先增加tests/test_meaningful_new40.py并观察缺少registry时失败，再写实现。
- 各core输入Local[N,256]、evidence[N,4,512]、active[N,4]、availability[N,3]，返回同维。无随机forward、跨话语状态或辅助loss；内层求导支持eval上下文。
- 用tests.test_meaningful_input复用共享模板：零初始化一致、inactive污染不泄漏、padding/首句、有限梯度及更新。
- 用tests.test_meaningful_block_integration验证真实模型参数/RNG/train/eval初始预测一致。遇到具体失败才补定向检查，不为每方法增加一般审查轮次。
- 提供select预留不超过剩余20个训练名额；不自动启动40训练。沿用原snapshot/CUDA readiness/数据/基线hash/GPU UUID约束。
- 统计真实参数量，区分CPU验证与尚未执行的biggpu真实数据CUDA检查、训练。物理GPU4禁用，总训练上限60保持。
- 检查diff，只暂存本任务文件，按Lore提交并推送github当前分支，不推origin、不提交权重或缓存。

验证命令：

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_new40
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_input tests.test_meaningful_block_integration
```

验收：40个真实工厂和非占位核心，原入口能构造；测试通过只是代码证据，不是性能证明。某方法若不能保留声明机制，换经过核实的新候选，不能静默简化凑数。
