# Flat / Relation Near-only Splice

INTERNAL DIAGNOSTIC ONLY

MOSI seed66；仅使用已保存预测，无训练、无新增推理、无权重修改。
固定abs(original Flat Local-only prediction)<=0.25时使用Relation full，否则保留Flat full。
Local-only是原Audit关闭Memory读值的预测；same/opposite和gold label绝不参与切换。
未扫描threshold。输入沿用每rate BEST Test-oracle，因此不是正式论文成绩或已验证部署方案。
沿用原ID及availability、label!=0过滤、pred>0分类；先按rate算W-F1再平均。

|Rate|Flat|Splice|Δpp|纠错|致错|N|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.205|88.220|+0.015|4|4|656|
|0.1|86.507|86.664|+0.158|4|3|656|
|0.2|83.187|83.325|+0.138|11|10|656|
|0.3|80.763|81.146|+0.383|10|7|656|
|0.4|80.827|80.571|-0.256|6|8|656|
|0.5|77.494|78.066|+0.572|16|13|656|
|0.6|75.790|77.073|+1.283|15|7|656|
|0.7|75.773|75.093|-0.680|8|13|656|
|8-rate mean|81.068|81.270|+0.202|74|65|5248|
|High .5/.6/.7|76.352|76.744|+0.392|39|33|1968|

|固定四格|Flat|Splice|Δpp|纠错|致错|N|
|---|---:|---:|---:|---:|---:|---:|
|near + same|81.987|81.336|-0.651|38|41|448|
|near + opposite|38.357|48.748|+10.391|31|15|229|
|far + same|89.664|89.664|+0.000|0|0|2728|
|far + opposite|70.906|70.906|+0.000|0|0|1419|

纠错/致错均相对于原Flat full。计数为跨rate重复出现次数，不是独立样本数。
固定四格沿用原分组，排除首句及任一端中性的相邻对；不是全测试集。
near两格保留Relation结果；far两格逐样本严格等于Flat。没有依据分组标签切换。
本结果只是固定阈值的事后拼接诊断，单seed且沿用Test-oracle checkpoint，不证明可训练单模型或多seed收益。
运行：python experiments/osram_current_history_relation_20261003/near_splice.py
