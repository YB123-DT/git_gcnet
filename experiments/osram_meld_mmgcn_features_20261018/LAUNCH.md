# 已验证的远程启动

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC — RUNNING, NOT FINAL RESULTS.

- 服务器 biggpu；原环境 `/data2/yb/conda_envs/cmerc-repro/bin/python`。
- 启动脚本固定于源码提交 `9c9ec92ae82927e5970258e81b794dc2dc959377`。
- 复用 worker SHA256：`66fc911607d06ebc98b7e7ab407aaf786185586eebcd8cb2ac4ab4e6b46a48d1`。
- 不可变数值源码：`737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`。
- 远程目录：`/home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018`。
- tmux：`meld_mmgcn_flat66_1018`；日志：上述目录的 `pipeline.log`。
- 正式任务 PID：`2109283`；开始 Unix 时间 `1792283414.782721`。
- 健康物理 GPU5：`GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`，进程内 cuda:0。
- 启动前 GPU5 约 32GB 空闲；输出所在 `/home` 约 130GB 空闲。
  GPU4 未使用，GPU6/7 上他人的任务未停止、暂停或修改。
- 转换数据传输后 SHA256：`d98f81538c1a2932ec3b959df6d02cf54930cf61063353174d284c9c1b8facd0`。
- 配置传输后 SHA256：`47dd72399dcbcf2615520c3be64490422516c2165ebc6c3ac506782c8d0c9faa`。

## 实际验证到哪一步

复用 worker 的真实三 batch 前向/反向/Adam 更新已完成，独立 smoke 输出：
`smoke/Flat_MMGCN_seed66/SUMMARY.json`。耗时 7.438 秒，峰值 allocated
593,329,664 bytes，Loss 和梯度有限，分类器权重确实更新。
这不是情感效果证据；正式训练是另一进程、重新设置原 seed。

正式训练已完成并提交至少前 2 个 epoch，进程运行中，`history.json` 与完整
`last_training.pt` 已存在。第一轮含 33 个 optimizer step、1038 段训练对话；
JEPA loss 为 0，cyclic 八率覆盖有效；rate=.7 实际缺失比例为原 GCNet 的 2/3。
已确认第一轮八率选点 artifact 每率均含 2610 条有效 test 样本，没有全空 mask。
Torch1.13 的 UUID RNG 枚举兼容修复由现有 worker 执行，训练恢复状态未报错。

有效维度 A/T/V 为 300/600/342，模型参数 13,181,403。相对 ECERC 模型
13,290,795 少 109,392，恰为 Text projector 的输入 Linear/LayerNorm 维度变化；
其余 Flat、Memory 和超参数没有调小。

可选 torch-sparse 导入仍有既有 warning；实际 OSRAM 三 batch 与正式训练运行
通过。没有安装依赖或改动 cmerc-repro 环境。

## 正式运行命令

通过 tmux 持久运行，pipeline 使用独占锁防重复提交：

```bash
tmux new-session -d -s meld_mmgcn_flat66_1018 \
  'bash /home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018/code/9c9ec92ae82927e5970258e81b794dc2dc959377/run.sh > /home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018/pipeline.log 2>&1'
```

参数：seed66、50 epoch、batch32、原 constant LR=.001、Flat1600/Local Skip、
gcnet-random-mask-v1、ACC-selected Test-oracle；只更换预提取特征。
**不同时采用上一轮余弦配置，避免把数据来源和训练设置混为一项变化。**

## 后续核验入口

正式输出：`runs/Flat_MMGCN_seed66`。结束时必须同时核验 SUMMARY complete、
50 个 history epoch、八个 BEST/选点预测与 last_training 状态，才能报告训练完成。
worker 会从 ACC-selected 同一份预测计算 UA/W-F1；不会通过新随机推理补指标。
当前启动记录不是最终分数，不声称更换特征有效；不自动追加种子或调参。
