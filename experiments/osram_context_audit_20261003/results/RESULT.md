# Context Audit — MOSI original cfg84 Flat

Offline reuse only:seed66/67/68 ×8rates, same checkpoint/mask/upstream per pair.
Local-only means alpha0 at the original Flat historical inputs; Local adapter and skip remain.
It is NOT a separately trained Local-only model. Memory calculations run unchanged in source inference.
G=(y-local)^2-(y-memory)^2. MSE includes neutral labels; polarity excludes label0, threshold prediction>0.
Near-zero:abs(G)<=1e-6. Mean G and percentages use rate-macro then seed-macro.
Counts pool repeated seed/rate exposures, not independent utterances. Empty group/rate cells excluded.
Internal Test-oracle label-assisted diagnosis; no learned selector, new fitting, or new inference.

|Group|N exposures|Mean G|MSE help %|MSE harm %|Near-zero %|Local wrong→Memory right|Local right→Memory wrong|Both right|Both wrong|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|all:ALL|16464|0.16448|51.67|43.81|4.52|1331|689|11369|2355|
|pattern:ATV|6157|0.03789|50.49|45.63|3.88|84|61|5065|679|
|position:first|744|0.00000|0.00|0.00|100.00|0|0|497|127|
|position:2-5|2976|0.04160|48.25|51.75|0.00|179|124|1954|575|
|position:6+|12744|0.20278|55.48|44.52|0.00|1152|565|8918|1653|
|local_margin:1+|6542|0.04172|47.24|47.05|5.71|17|7|5970|409|
|local_margin:.25-1|6956|0.21054|53.31|42.93|3.76|600|287|4261|1418|
|local_margin:0-.25|2966|0.31382|56.91|39.27|3.82|714|395|1138|528|
|label_strength:strong>1|10344|0.27972|55.21|40.38|4.41|849|336|8074|1085|
|label_strength:weak<=1|5400|-0.01516|47.61|49.28|3.11|482|353|3295|1270|
|label_strength:neutral|720|-0.14387|31.25|52.08|16.67|0|0|0|0|
|pattern:AV|1640|0.31203|57.31|37.85|4.84|275|156|788|345|
|pattern:AT|1584|0.03458|48.64|46.73|4.63|39|28|1268|175|
|pattern:TV|1645|0.01749|47.56|47.65|4.79|15|21|1358|182|
|pattern:V|1881|0.36556|55.97|39.83|4.20|328|182|853|446|
|pattern:A|1771|0.59637|60.46|35.51|4.03|554|224|593|323|
|pattern:T|1786|0.03930|48.45|45.92|5.63|36|17|1444|205|

Patterns describe CURRENT availability under random missing, not persistent whole-conversation masks.
Local-margin groups are descriptive and not calibrated uncertainty. Label-strength groups require labels.
MSE improvement does not imply polarity correction, nor does MSE worsening imply wrong classification.
These observations identify strata, not semantic utterance types or a deployable rule for memory usage.
