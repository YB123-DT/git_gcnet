# Offline Relation comparison

INTERNAL DIAGNOSTIC ONLY

Per-rate BEST Test-oracle checkpoints; seed67 screening, not a formal paper result.
No training/inference here. Label != 0; prediction > 0. W-F1 in percent; differences in percentage points.
Fixed cells use OLD Flat Local-only predictions, not Relation-model predictions.
Same/opposite means adjacent utterance label polarity, not a true emotion transition.

|Rate|Flat full|Relation full|Delta|Corrections|Harms|N|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.321|87.865|-0.456|11|14|656|
|0.1|86.103|86.061|-0.042|14|14|656|
|0.2|84.872|84.308|-0.564|32|34|656|
|0.3|80.067|80.130|+0.063|49|50|656|
|0.4|77.115|77.307|+0.192|27|28|656|
|0.5|76.693|76.271|-0.422|23|26|656|
|0.6|75.536|75.536|+0.000|39|39|656|
|0.7|75.739|76.671|+0.932|30|25|656|
|8-rate mean|80.556|80.519|-0.037|225|230|5248|
|High mean|75.990|76.160|+0.170|92|90|1968|

|Fixed boundary|Adjacent relation|Old Local-only|Flat full|Relation full|Delta vs Flat|Corrections|Harms|N exposures|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|near|same|57.397|79.705|79.439|-0.267|51|48|499|
|near|opposite|58.091|43.231|48.972|+5.741|59|37|313|
|far|same|85.236|88.181|87.658|-0.524|61|75|2677|
|far|opposite|76.184|74.938|74.524|-0.414|42|47|1335|

Corrections/harms compare Flat full → Relation full on exactly the same samples.
W-F1 is calculated per seed/rate then macro-averaged; never pooled across rates.
Counts sum repeated rate exposures, not independent utterances. No significance claim from one seed.
No-ID trainer artifacts retain canonical evaluation-loader order; exact label and availability arrays are checked against the audited original artifacts.
