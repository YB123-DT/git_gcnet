# Offline Relation comparison

INTERNAL DIAGNOSTIC ONLY

Per-rate BEST Test-oracle checkpoints; seed68 screening, not a formal paper result.
No training/inference here. Label != 0; prediction > 0. W-F1 in percent; differences in percentage points.
Fixed cells use OLD Flat Local-only predictions, not Relation-model predictions.
Same/opposite means adjacent utterance label polarity, not a true emotion transition.

|Rate|Flat full|Relation full|Delta|Corrections|Harms|N|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.730|87.742|-0.988|11|17|656|
|0.1|84.926|84.993|+0.067|25|24|656|
|0.2|82.235|82.237|+0.002|30|31|656|
|0.3|82.169|81.374|-0.795|28|33|656|
|0.4|79.047|79.215|+0.168|34|34|656|
|0.5|77.792|78.261|+0.468|48|45|656|
|0.6|76.220|75.558|-0.661|59|62|656|
|0.7|69.307|68.847|-0.461|52|50|656|
|8-rate mean|80.053|79.778|-0.275|287|296|5248|
|High mean|74.440|74.222|-0.218|159|157|1968|

|Fixed boundary|Adjacent relation|Old Local-only|Flat full|Relation full|Delta vs Flat|Corrections|Harms|N exposures|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|near|same|57.722|79.716|74.468|-5.248|64|89|683|
|near|opposite|60.160|44.557|49.678|+5.121|77|59|391|
|far|same|86.050|88.778|88.764|-0.014|70|68|2493|
|far|opposite|75.986|75.099|74.094|-1.005|42|54|1257|

Corrections/harms compare Flat full → Relation full on exactly the same samples.
W-F1 is calculated per seed/rate then macro-averaged; never pooled across rates.
Counts sum repeated rate exposures, not independent utterances. No significance claim from one seed.
No-ID trainer artifacts retain canonical evaluation-loader order; exact label and availability arrays are checked against the audited original artifacts.
