# M01–M40 代码交付（中文）

INTERNAL DIAGNOSTIC ONLY。用户指定40项已实现，尚未训练；完整中文记录见[RESULT.md](RESULT.md)。

后续状态：启动授权已恢复，40项接入biggpu持久补位队列。
见`LAUNCH.md`和`LAUNCH_STATUS.json`；以下是代码交付时的历史记录。

37项R保留原Flat并加零初始化残差，M11/M13仅调制Adapter输入，M30仅替换Adapter第一层LN。
不改Memory、Query、任务头、loss或mask；Local Skip保留，inactive Gap和padding安全屏蔽。
M30不是LN等价初始化。39项R/I通过整模初始化/RNG检查。

另外22项实现也已接入，但合计62个入口不能当作62个全新机制；120目标尚未完成。
CPU检查与参数量见`VERIFICATION.json`和`configs/SUMMARY.json`。
配置继承真实cfg84 seed66，仅改变模块开关。新实验和GPU smoke没有启动，旧训练保留。
历史Test-oracle协议仅供内部诊断，不能作为正式论文结果。

出处与适配限制记在代码；历史重复不计新，未知项保持待审查。无性能提升声明。
