# Frozen history drift — original cfg84 seed66

No training or model changes. Validation/test separate; inherited per-rate Test-oracle checkpoints.
Only same-current/different-strict-history anchors. Base/Gap effective first512dimensions.
Zero-vector cosine excluded with valid counts, not replaced by0or1.

Main numbers are unweighted macro means over rates with eligible anchors/nonzero vectors. Gap has no
active slots at rate0; its macro therefore excludes that rate. Per-metric rate counts retained in JSON.
Counts sum rate exposures,
not independent samples. Flip% excludes neutral labels and uses original prediction>0 threshold.
Local max is the largest absolute coordinate difference. Different layers use different scales;
these numbers alone cannot establish excessive amplification or harmful drift.

## validation: nominal deletion interventions

|Delete|N anchors|Rates|Local max|Base cosine drift|Gap cosine drift|Hidden cosine drift|Prediction shift|Flip%|Correct→wrong|Wrong→correct|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A|1032|8|1.907e-06|0.00476|0.01655|0.00001|0.00745|0.10965|1|0|
|T|978|8|1.907e-06|0.05193|0.03343|0.00036|0.06282|2.29426|10|12|
|V|965|8|1.907e-06|0.00689|0.00614|0.00003|0.01668|0.51489|3|1|
|mixed|1097|8|1.907e-06|0.04542|0.03082|0.00026|0.05213|1.86439|8|11|

### validation: matched A/T/V anchors

|Delete|N anchors|Prediction shift|Flip%|Mean prior deleted bits|
|---|---:|---:|---:|---:|
|A|494|0.00712|0.00000|2.19287|
|T|494|0.05104|1.86701|2.31052|
|V|494|0.01781|0.95081|2.45295|

### validation: nearest deletion distance, mixed intervention

|Distance|N anchors|Prediction shift|Flip%|Mean prior deleted bits|
|---|---:|---:|---:|---:|
|1|318|0.10298|3.53744|4.86635|
|2|212|0.06138|1.36040|4.76151|
|3-4|248|0.02870|0.76941|4.31375|
|5+|319|0.01007|0.63595|3.22132|

## test: nominal deletion interventions

|Delete|N anchors|Rates|Local max|Base cosine drift|Gap cosine drift|Hidden cosine drift|Prediction shift|Flip%|Correct→wrong|Wrong→correct|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A|2898|8|1.907e-06|0.00418|0.01152|0.00001|0.00769|0.38578|6|3|
|T|2795|8|1.907e-06|0.05194|0.04872|0.00045|0.06894|3.58689|52|40|
|V|2871|8|1.907e-06|0.00677|0.01301|0.00004|0.01821|0.32158|2|6|
|mixed|3005|8|1.907e-06|0.04304|0.04382|0.00033|0.05943|2.61594|46|30|

### test: matched A/T/V anchors

|Delete|N anchors|Prediction shift|Flip%|Mean prior deleted bits|
|---|---:|---:|---:|---:|
|A|1379|0.00507|0.15574|2.25475|
|T|1379|0.05702|2.99810|2.37841|
|V|1379|0.01554|0.10163|2.49129|

### test: nearest deletion distance, mixed intervention

|Distance|N anchors|Prediction shift|Flip%|Mean prior deleted bits|
|---|---:|---:|---:|---:|
|1|962|0.11267|5.64645|4.79833|
|2|631|0.06498|3.21727|4.81918|
|3-4|682|0.03392|1.60367|4.56801|
|5+|730|0.01165|0.24298|4.35726|

## Limits

Nominal modes can have different eligible anchors/deletion counts; matched A/T/V table controls
current sample/rate but not number or timing of prior deletions. Mixed prefix group breakdown,
individual Gap drifts, pooled statistics and current-pattern strata are in SUMMARY.json/CSV.
Distance means nearest prior deletion under repeated deletions, not a single isolated perturbation.
Cosine can stay unchanged despite magnitude drift; norm differences are retained in the tables.
Historical information differs genuinely; flips may correct or harm. No labels used to fit or select.
Single seed and Test-oracle origins preclude formal generalization/significance claims.
