# Offline Relation comparison

INTERNAL DIAGNOSTIC ONLY

Per-rate BEST Test-oracle checkpoints; seed66 screening, not a formal paper result.
No training/inference here. Label != 0; prediction > 0. W-F1 in percent; differences in percentage points.
Fixed cells use OLD Flat Local-only predictions, not Relation-model predictions.
Same/opposite means adjacent utterance label polarity, not a true emotion transition.

|Rate|Flat full|Relation full|Delta|Corrections|Harms|N|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.205|88.071|-0.134|13|14|656|
|0.1|86.507|85.912|-0.595|16|20|656|
|0.2|83.187|82.702|-0.485|21|24|656|
|0.3|80.763|80.506|-0.257|31|32|656|
|0.4|80.827|80.366|-0.460|28|32|656|
|0.5|77.494|77.002|-0.492|48|51|656|
|0.6|75.790|75.267|-0.523|46|50|656|
|0.7|75.773|74.786|-0.988|40|48|656|
|8-rate mean|81.068|80.576|-0.492|243|271|5248|
|High mean|76.352|75.685|-0.667|134|149|1968|

|Fixed boundary|Adjacent relation|Old Local-only|Flat full|Relation full|Delta vs Flat|Corrections|Harms|N exposures|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|near|same|52.265|81.987|81.336|-0.651|38|41|448|
|near|opposite|56.885|38.357|48.748|+10.391|31|15|229|
|far|same|80.405|89.664|87.895|-1.769|75|124|2728|
|far|opposite|74.797|70.906|71.678|+0.772|83|72|1419|

Corrections/harms compare Flat full → Relation full on exactly the same samples.
W-F1 is calculated per seed/rate then macro-averaged; never pooled across rates.
Counts sum repeated rate exposures, not independent utterances. No significance claim from one seed.
No-ID trainer artifacts retain canonical evaluation-loader order; exact label and availability arrays are checked against the audited original artifacts.
