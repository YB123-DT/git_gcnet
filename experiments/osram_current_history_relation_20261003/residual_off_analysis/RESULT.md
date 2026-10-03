# Relation residual-off decomposition

INTERNAL DIAGNOSTIC ONLY

Per-rate BEST Test-oracle checkpoints; one seed. No retraining, threshold search, or label-conditioned inference.
F = original Flat; R-off = trained Relation checkpoint with only residual forced to zero; R-on = same checkpoint with residual enabled.
W-F1 (%), nonzero labels only, prediction > 0. Fixed near/far groups use original Flat Local-off predictions at 0.25; same/opposite is reporting only.
Scores are computed within each seed/rate then macro-averaged; counts sum rate exposures, not independent samples.

|Rate / group|F|R-off|R-on|Off−F|On−Off|On−F|N|
|---|---:|---:|---:|---:|---:|---:|---:|
|rate 0.0|88.205|88.234|88.071|0.029|-0.163|-0.134|656|
|rate 0.1|86.507|85.921|85.912|-0.586|-0.009|-0.595|656|
|rate 0.2|83.187|82.585|82.702|-0.601|0.116|-0.485|656|
|rate 0.3|80.763|80.373|80.506|-0.390|0.133|-0.257|656|
|rate 0.4|80.827|80.085|80.366|-0.741|0.281|-0.460|656|
|rate 0.5|77.494|76.896|77.002|-0.598|0.106|-0.492|656|
|rate 0.6|75.790|75.181|75.267|-0.608|0.086|-0.523|656|
|rate 0.7|75.773|73.610|74.786|-2.163|1.175|-0.988|656|
|8-rate mean|81.068|80.361|80.576|-0.707|0.216|-0.492|5248|
|High mean|76.352|75.229|75.685|-1.123|0.456|-0.667|1968|
|near + same|81.987|80.094|81.336|-1.893|1.242|-0.651|448|
|near + opposite|38.357|48.205|48.748|9.848|0.543|10.391|229|
|far + same|89.664|87.829|87.895|-1.835|0.067|-1.769|2728|
|far + opposite|70.906|71.781|71.678|0.875|-0.103|0.772|1419|

|Rate / group|Off−F corrections / harms|On−Off corrections / harms|On−F corrections / harms|
|---|---:|---:|---:|
|rate 0.0|13 / 13|1 / 2|13 / 14|
|rate 0.1|16 / 20|1 / 1|16 / 20|
|rate 0.2|20 / 24|4 / 3|21 / 24|
|rate 0.3|31 / 33|2 / 1|31 / 32|
|rate 0.4|30 / 36|4 / 2|28 / 32|
|rate 0.5|48 / 52|3 / 2|48 / 51|
|rate 0.6|43 / 48|6 / 5|46 / 50|
|rate 0.7|35 / 51|10 / 2|40 / 48|
|8-rate mean|236 / 277|31 / 18|243 / 271|
|High mean|126 / 151|19 / 9|134 / 149|
|near + same|37 / 45|5 / 0|38 / 41|
|near + opposite|28 / 15|5 / 2|31 / 15|
|far + same|70 / 121|9 / 7|75 / 124|
|far + opposite|84 / 72|7 / 8|83 / 72|

The W-F1 difference decomposition is an algebraic identity on matched samples, not independent causal effects.
Corrections and harms are separately recomputed for each paired comparison and are NOT individually additive across the two stages.
R-off−F describes the trained original-path change; R-on−R-off is the conditional inference-time effect of enabling the residual at the trained checkpoint.
It does not isolate what training the original Flat alone under an identical optimization trajectory would have produced.
Adjacent same/opposite polarity is not a true emotion transition; no mechanism or significance claim is made.
