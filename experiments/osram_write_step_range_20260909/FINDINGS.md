# 写入步长的粗粒度拐点定位

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

## 结论

两套数据的五率均值都在已测试的 eta=0.6 达到峰值；邻近采样点是
0.4 和 0.8。继续降到 0.2 或 0 后明显下降。因此不是“write 越弱越好”。
这是冻结 checkpoint 的采样曲线，不是连续最优值，也不是训练后收益。
本轮不再加密搜索，不启动重新训练。

| eta | IEMOCAP-4 W-F1 % | MOSI W-F1 % |
|---|---:|---:|
| 1.00 | 81.624 | 80.066 |
| 0.95 | 81.794 | 80.137 |
| 0.90 | 81.867 | 80.145 |
| 0.80 | 82.164 | 80.201 |
| 0.60 | **82.358** | **80.350** |
| 0.40 | 82.159 | 80.325 |
| 0.20 | 80.714 | 79.980 |
| 0.00 | 66.374 | 77.974 |

五 seeds=66–70；五 rates=0/0.1/0.3/0.5/0.7。
先在每个 seed 内平均五率，再对五 seeds 等权平均。不是八率均值。
完整逐 rate、逐 seed、标准差见 SUMMARY.md 和 CSV。

## 稳定性与规律边界

- IEMOCAP-4：0.6 相对 1.0 +0.7340 pp，5/5 seeds、5/5 rates 为正。
  相对 0.8 +0.1943 pp，4/5 seeds 为正；相对 0.4 +0.1990 pp，4/5 为正。
  各 seed 的采样最佳点依次为 0.6/0.8/0.4/0.6/0.6。
- MOSI：0.6 相对 1.0 +0.2837 pp，4/5 seeds、4/5 rates 为正。
  miss=0.5 仍略低于 Reference（76.309 vs 76.322）。
  相对 0.8 +0.1494 pp，4/5 seeds 为正；相对 0.4 仅 +0.0251 pp，3/5 为正。
  因此 MOSI 的 0.4–0.6 更像宽平台，不能宣称 0.6 显著优于 0.4。
  各 seed 的采样最佳点依次为 0.6/0.6/0.4/1.0/0.95。
- eta=0 下 memory 从零开始且从不写入，Base/Gap 为零。IEMOCAP-4
  尤其依赖当前已训练模型的上下文路径；MOSI 在高缺失时也明显受损。
  这不是重新训练的 Local-only 对照，不能据此定量判断上下文的普遍价值。

## 不能把 F1 改善解释为旧 association 拟合更好

下面是 miss=0.7，先按 modality 内统计，再对三个 modality 和五 seeds
等权平均的相对误差；不是按 utterance/head 数量加权。E_old 为实际读取时
err_decay，排除 NO_HISTORY；E_new 为当前 observed slot 的 post-write err_after。

| 数据集 | eta | E_old | E_new |
|---|---:|---:|---:|
| IEMOCAP-4 | 1.0 | 0.207279 | 0.001860 |
| IEMOCAP-4 | 0.6 | 0.395763 | 0.319430 |
| IEMOCAP-4 | 0.4 | 0.516858 | 0.472290 |
| MOSI | 1.0 | 0.132166 | 0.001098 |
| MOSI | 0.6 | 0.271848 | 0.202082 |
| MOSI | 0.4 | 0.381113 | 0.326193 |

两种拟合误差都随减弱 write 增大，F1 却先升后降。
当前证据支持“冻结模型对写入强度存在任务层面的敏感区间”，不支持
“旧 association 误差降低导致 F1 提升”。平滑、幅度变化或推理正则化
只能作为待验证解释。当前 post-write 只影响未来读取，不能和当前分类误差
直接建立因果联系。是否消除 train-test mismatch 后仍有效，尚未验证。

## 协议与来源

- 旧四点直接继承 osram_write_step_grid_20260909；其中 IEMOCAP-4
  Reference/Fixed0.9 继续追溯到 osram_write_intervention_20260909/full5。
- 新四点是在看到旧结果之后锁定的事后扩展，不伪称八点全部预注册。
- 200 个新增 evaluation cells + 200 个继承 cells，共 400 cells。
- 每 seed 使用原有单一 best.pt，原 checkpoint 由八率均值 Test-oracle
  选出；本轮不按 eta/rate 重新选择 epoch，不进行 optimizer step。
- IEMOCAP-4 epochs：66→33、67→54、68→38、69→30、70→64。
  MOSI epochs：66→54、67→49、68→30、69→61、70→50。
- 汇总已通过 checkpoint/config/epoch/selection 元数据和全模式 mask hash
  一致性检查；评估器验证 state_dict 全部 tensor 未变化。
- CPU evaluation-only：biggpu，每进程 2 torch threads，每数据集并行 5 seeds；
  IEMOCAP-4 完成后运行 MOSI。没有 GPU 训练。
- 所有 10 个新评估进程 exit=0；204 个模型/诊断测试、18 个统计测试通过。
  一条既有 PyG deprecation warning；无测试失败。
- 正式模型 osram.py/model.py/train_gcnet.py 未修改。本次仅增加外置干预
  的四个固定步长，以及汇总器和对应测试。
- 600 个完整压缩 raw 文件保留本地和远端，Git 保存 RAW_MANIFEST.csv
  的 SHA256、所有 metadata/summary 和分析结果，避免把原始大文件混入代码。
  远端根目录：/data2/yb/remote_experiments/osram_write_step_range_20260909/。
- 原始记录逐行复核：IEMOCAP-4 3,935,920 行、MOSI 2,160,144 行，
  共 6,096,064 行；5,520 组指标 count/mean/min/max 与摘要一致。
  所有数值有限，observed target 均与可见 mask 一致，fit_gain 恒等式偏差为 0。
  NO_HISTORY 分别 5,040 / 5,008 条，未混入 retention error。
  eta=0 的全部 write applied_norm=0；有效旧/新 probe error=1，damage=0。

## 复核入口

```bash
python3 experiments/osram_write_step_range_20260909/summarize_range.py
```

本轮找到的区域是 **0.4–0.8 内的中等写入强度，采样均值峰值 0.6**。
不继续密集挑选 Test 参数；下一阶段若进行训练验证，必须与本轮冻结干预
结果分开，且不能把本轮 Test 搜索当作独立正式证据。
