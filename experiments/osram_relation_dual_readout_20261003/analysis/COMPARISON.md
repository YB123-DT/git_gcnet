# Offline Relation comparison

INTERNAL DIAGNOSTIC ONLY

Per-rate BEST Test-oracle checkpoints; seed66 screening, not a formal paper result.
No training/inference here. Label != 0; prediction > 0. W-F1 in percent; differences in percentage points.
Fixed cells use OLD Flat Local-only predictions, not Relation-model predictions.
Same/opposite means adjacent utterance label polarity, not a true emotion transition.

|Rate|Flat full|Relation full|Delta|Corrections|Harms|N|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.205|87.466|-0.739|12|17|656|
|0.1|86.507|86.572|+0.065|24|24|656|
|0.2|83.187|82.828|-0.359|28|31|656|
|0.3|80.763|80.591|-0.172|38|40|656|
|0.4|80.827|80.591|-0.236|39|42|656|
|0.5|77.494|77.018|-0.476|33|38|656|
|0.6|75.790|75.093|-0.696|36|41|656|
|0.7|75.773|75.036|-0.737|46|49|656|
|8-rate mean|81.068|80.649|-0.419|256|282|5248|
|High mean|76.352|75.716|-0.636|115|128|1968|

|Fixed boundary|Adjacent relation|Old Local-only|Flat full|Relation full|Delta vs Flat|Corrections|Harms|N exposures|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|near|same|52.265|81.987|79.848|-2.139|34|44|448|
|near|opposite|56.885|38.357|54.887|+16.529|43|17|229|
|far|same|80.405|89.664|88.005|-1.659|76|122|2728|
|far|opposite|74.797|70.906|71.597|+0.691|81|71|1419|

Corrections/harms compare Flat full → Relation full on exactly the same samples.
W-F1 is calculated per seed/rate then macro-averaged; never pooled across rates.
Counts sum repeated rate exposures, not independent utterances. No significance claim from one seed.
No-ID trainer artifacts retain canonical evaluation-loader order; exact label and availability arrays are checked against the audited original artifacts.
