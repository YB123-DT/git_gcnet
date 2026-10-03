# Relation residual 置零诊断

INTERNAL DIAGNOSTIC ONLY

完整逐率、固定四格及三组纠错/致错表见
[residual_off_analysis/RESULT.md](residual_off_analysis/RESULT.md)。

## 结果

|范围|原Flat F|Relation关残差 R-off|Relation开残差 R-on|Off−F|On−Off|On−F|
|---|---:|---:|---:|---:|---:|---:|
|八率平均|81.068|80.361|80.576|−0.707|+0.216|−0.492|
|高缺失平均|76.352|75.229|75.685|−1.123|+0.456|−0.667|
|near + same|81.987|80.094|81.336|−1.893|+1.242|−0.651|
|near + opposite|38.357|48.205|48.748|+9.848|+0.543|+10.391|
|far + same|89.664|87.829|87.895|−1.835|+0.067|−1.769|
|far + opposite|70.906|71.781|71.678|+0.875|−0.103|+0.772|

W-F1%，差值为百分点；先各rate计算再宏平均。显示值经四舍五入，未舍入值满足差值恒等式。
本次沿用原Flat Local-only（关闭Memory读值）的预测定义near/far；
same/opposite仅为离线分组，不是推理输入。

在这些已选checkpoint上，总体损失主要表现为Relation联合训练后原路径的差异，
而非打开residual本身带来的总体损失。打开residual累计纠错31、致错18，
高缺失纠错19、致错9。它补回部分性能，但没有恢复到原Flat。

near+opposite的改善绝大部分在关闭residual后仍然保留（+9.848），
直接打开residual仅再增加+0.543；不能把先前这格+10.391主要归功于推理时关系残差。
near+same中，打开残差带来5次纠错、0次致错，但仍未追回原路径差异。
far+same在关闭残差时已经下降−1.835，开启仅补回+0.067。

Off−F不是“只改变一个参数”的训练因果实验：两者经历不同联合优化轨迹，且分别使用
各自原先选定的per-rate BEST epoch。它包含整个原路径（encoder、Memory、Flat、head）
以及checkpoint选点差异，不能单独归因于某个Memory参数。
On−Off则是同一个Relation checkpoint下关闭/打开残差的条件性推理效应。
W-F1差值的可加性是代数恒等式；三组纠错/致错必须各自计算，不能逐项相加。
这些结果不证明真实emotion shift、独立因果机制或多seed稳定收益。

## 执行与完整性

- 不训练、不改权重、不扫threshold、不更换或重新选择checkpoint。
- 源权重：biggpu `/data2/yb/remote_experiments/osram_current_history_relation_20261003/seed_66`。
- 新诊断：相同目录下`residual_off_results_v2`；GPU5，PID630535，代码`2eb8f83`。
- 干预位置：`relation_block` forward-output hook，仅返回`zeros_like(residual)`。
  Base/Gap、Memory写入、Local Skip、Flat adapter、emotion_norm和任务头均未改动。
- 每rate先回放R-on校验旧预测，再做R-off；模型eval/no_grad，无optimizer。
- 八率on/off的Local、Base、Gap、availability、umask及Flat pre-norm anchor哈希完全一致。
- 模型state_dict逐tensor检查未变；源checkpoint文件SHA256前后未变。
- 八率R-off residual norm均严格为0；labels、availability及mask hash一致。
- 回放R-on与旧保存预测存在FP32微小误差，八率最大绝对误差4.76837158203125e-7，
  全部通过atol1e-6/rtol0及**极性逐样本完全一致**检查，W-F1和纠错统计不受影响。
- 首次逐位相等断言在rate0失败（最大误差2.3841858e-7），在R-off推理前停止。
  失败日志和provenance保留于`residual_off_failed_attempt/`。只调整回放验证容差，
  未改变模型、精度、数据、权重或推理协议；具体底层浮点差异来源未进一步定位。
- 4项单元测试通过：输出置零且参数/输入不变、拒绝超限误差及任意极性变化、
  差值恒等式、纠错/致错不可逐项相加、rate宏平均。
- 分析读取固定旧分组，校验所有来源hash，不覆盖旧预测。没有新增其它实验。

命令（远程code快照目录）：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_current_history_relation_20261003/residual_off.py \
--source /data2/yb/remote_experiments/osram_current_history_relation_20261003/seed_66 \
--output /data2/yb/remote_experiments/osram_current_history_relation_20261003/residual_off_results_v2 \
--commit 2eb8f83
```

离线汇总：
`python experiments/osram_current_history_relation_20261003/residual_off_analyze.py`。
输出目录需尚不存在，以避免覆盖既有诊断。
