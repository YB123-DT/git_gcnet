# Current–History Relation Block：MOSI seed66 初筛

**INTERNAL DIAGNOSTIC ONLY**

结论：实现及正确性检查通过；本轮性能未通过初筛，不建议直接扩大 multi-seed。
八率均值下降0.492个百分点，高缺失下降0.667；八个rate均低于原Flat。
固定near+opposite子组改善，不能抵消全测试集下降。

## 版本、协议与运行

- 本轮修改前的Flat实现commit：`7d56881f13e50b98a65a8fc9ca19f31222b39a55`。
- Relation实现commit：`7777d2d`；当前分支`feature/osram-uniform-forced-text`。
- 原Flat历史checkpoint：biggpu上
  `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66`。
  历史PROVENANCE未记录Git commit，不能把本轮修改前commit冒充其训练commit；
  原训练源码SHA、配置及来源链保存在本目录`reference/PROVENANCE.json`。
- 原Flat不重训。Relation从头一阶段联合训练，MOSI seed66、100epochs、batch32、
  Adam、lr0.001、weight decay1e-5、原MSE task loss，cyclic random missing0.0–0.7。
- 每rate按测试W-F1选BEST，**Test-oracle内部筛选，不是validation选模或正式论文成绩**。
  没有修改Memory读写、block write、Query或task head；没有新增辅助loss、Gate、
  JEPA、completion、双视图、InfoNCE或persistent mix。
- biggpu宿主GPU5，UUID`GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`，PID568195。
  04:28:47–04:36:56 UTC，完成100轮；200个optimizer steps/forward，JEPA loss和EMA steps均为0。
- 八个BEST权重保留于
  `/data2/yb/remote_experiments/osram_current_history_relation_20261003/seed_66`，不上传权重。
  配置、历史、预测、diagnostics和provenance已同步到`results/seed_66/`。

运行命令见[README.md](README.md)及完整[launch.json](launch.json)。

## 实现

保留`u_flat=local_skip(L)+emotion_adapter([L,Base,masked Gap-A/T/V])`。
只取历史读值的forward512维，投影`q=P_L(L), k_i=P_C(C_i)`至128维，
共享`P_C`；type embedding为16维。

```text
z_i = [q, k_i, q*k_i, abs(q-k_i), type_i]             # 528
r_i = Linear64(Dropout(GELU(Linear128(LayerNorm(z_i)))))
r   = sum(active_i * r_i) / max(1, sum(active_i))
h   = emotion_norm(u_flat + relation_out(r))         # 64→1600, zero-init
```

Dropout沿用cfg84的0.5。有效历史位置Base active，Gap仅在对应模态当前缺失时active。
输入、type拼接结果和输出均安全mask。用严格的此前valid-count识别首个有效utterance，
在最终有bias的输出投影之后再次屏蔽，确保它在训练后仍然没有relation residual。
padding同样严格零残差和零OSRAM hidden。原task head保持不变。
模块初始化和训练期额外Dropout隔离RNG，关闭开关恢复原路径。
这是历史关系表示残差，不是证据权重或Memory强度门控。

### 参数量 / 容量对照

|模型|总参数（全部可训练）|新增参数|
|---|---:|---:|
|原Flat|13,509,793|0|
|Pairwise Relation|13,789,441|279,648|
|Pooled residual MLP control|13,789,327|279,534|

Relation增加约2.07%。control为`[q;active-mean(k)]`256维，
LayerNorm→Linear238→GELU→Dropout→Linear64→相同零初始化输出。
没有乘积、绝对差或逐证据pairwise MLP；与Relation相差114参数。
**Control只实现和验证，未训练，因此本轮不能声称relation结构优于普通容量增加。**

## 全测试集结果

W-F1单位%，差值单位百分点；排除label=0，预测阈值>0。
每rate656个非中性utterance；均值是rate级W-F1的算术平均。

|Missing rate|Flat|Relation|Δ|Relation BEST epoch|
|---|---:|---:|---:|---:|
|0.0|88.205|88.071|−0.134|84|
|0.1|86.507|85.912|−0.595|84|
|0.2|83.187|82.702|−0.485|75|
|0.3|80.763|80.506|−0.257|83|
|0.4|80.827|80.366|−0.460|50|
|0.5|77.494|77.002|−0.492|83|
|0.6|75.790|75.267|−0.523|75|
|0.7|75.773|74.786|−0.988|86|
|8-rate mean|81.068|80.576|−0.492|—|
|High (.5/.6/.7)|76.352|75.685|−0.667|—|

全测试集跨rate累计纠错243、致错271，净少28次正确预测。
这不是独立样本计数，也不能从净计数直接换算W-F1差值。

## 固定四格：旧Flat full → Relation full

完全复用原`adjacent_results/utterances.csv`中seed66分组。
near/far由**原Flat Local-only（Memory读值关闭，非关闭Local）**预测定义：
abs(pred)<=0.25为near；相邻两端标签非零，按立即前一句同／异极性分组。
不跨conversation，不跳过中性标签，不用新模型定义分组；标签仅用于离线统计。

|固定组|旧Local-only|Flat full|Relation full|Δ vs Flat|纠错|致错|N次出现|
|---|---:|---:|---:|---:|---:|---:|---:|
|near + same|52.265|81.987|81.336|−0.651|38|41|448|
|near + opposite|56.885|38.357|48.748|+10.391|31|15|229|
|far + same|80.405|89.664|87.895|−1.769|75|124|2728|
|far + opposite|74.797|70.906|71.678|+0.772|83|72|1419|

各格先计算每rate W-F1，再宏平均；N和纠错/致错跨rate累加。
总4824次出现；四格不等于整个测试集，首句及含中性的相邻对不入表。
各组W-F1不能线性相加重建全测试集W-F1。

失败位置：near+same保留大部分原历史收益；near+opposite的确改善，但仍低于旧Local-only。
主要的净致错集中于样本更多的far+same（净增加49次错误），抵消了异极性组的改善。
这说明当前版本没有实现“局部改善同时提升全测试集”的目标。
**不能据此断言模型检测了冲突或真实emotion shift，也不能断言它只是削弱了所有Memory**：
本轮是全参数联合训练，未对训练后的模型做这种机制隔离。

## Diagnostics / 验证

每rate评测均记录686个valid位置和655个history-supported位置。
所选checkpoint的平均residual norm为6.159–7.343，平均逐样本residual/anchor norm比
约2.12%–2.29%；Base/各Gap relation norm均存于metrics/history。
这些只是内部量，不能自动解释为冲突、可靠性或情绪转折概率。

- V100上18项检查全部通过；CPU上额外7项通过，详见[VERIFICATION.md](VERIFICATION.md)。
- 初始完整模型train/eval输出逐值等于Flat；关闭开关与实际修改前源码一致。
- inactive Gap/unused backward half的NaN不泄漏；首句、padding严格屏蔽。
- 三步Adam后梯度有限，relation和原adapter/skip/query均更新，无冻结。
- 真实旧Flat checkpoint严格加载成功；8个新BEST存在且回读评测完成。
- 八率mask hash以及逐样本label/availability均与原基线一致；所有原config字段未改。
- 本地实现source SHA与运行provenance一致；history完整100轮；任务loss唯一、无额外forward。
- 离线审计保存输入hash且未覆盖原预测，详见`analysis/AUDIT.json`。

## 后续判断

**不建议当前版本直接继续multi-seed，也未自动启动其他训练。**
本轮只有一个seed、使用Test-oracle，无显著性或稳定提升结论。
保留这份负结果和固定四格变化；容量对照已准备但尚未提供性能证据。
没有因near+opposite单格提高就宣布方法成功，也没有据此重新调参重跑。
