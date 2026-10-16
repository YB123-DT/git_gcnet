# Corrected GCNet missing: ACC and UA, with evaluation stages separate

INTERNAL TEST-ORACLE DIAGNOSTIC.100epochs, equal fivefold/eight-rate averages,
seeds66/67/68. UA is macro class recall (4 or6 classes), not macro-F1.
No training or inference was run for this report.

## A. Per-rate ACC-selected Test-best: current primary reporting stage

All B6 UA values use the SAME selected epoch, mask and predictions as ACC.
Original Flat's matching selected-epoch UA was not retained; an em dash is
missing evidence, NOT zero. Do not fill it with fresh-mask retest UA.

| Classes | Seed | Flat ACC | Flat UA | B6 ACC | B6 UA |
|---|---:|---:|---:|---:|---:|
|4|66|79.198|—|79.408|79.655|
|4|67|79.097|—|79.155|79.474|
|4|68|79.487|—|79.298|79.564|
|4|Mean|79.261|—|79.287|79.564|
|6|66|61.175|—|61.637|59.819|
|6|67|62.046|—|61.746|60.392|
|6|68|61.653|—|61.818|60.147|
|6|Mean|61.625|—|61.734|60.119|

Flat ACC source: original metrics.selected_accuracy_by_rate. B6 source:
SUMMARY.json datasets.*.fivefold.seeds.*.mean, previously checked against
240 selected-epoch prediction artifacts. Original selected UA cannot be
recovered from scalar accuracy/weighted-F1/macro-F1 alone. The old original
Flat checkpoint-discarding runner did not retain its selected weights either;
no unrequested retraining is launched to recreate them.

## B. Original Flat: separately saved fresh-mask final retest

The following ACC/UA pairs are internally consistent, but NOT stageA's scores.
They are recomputed from the SAME saved final predictions per rate/fold,
without invoking the model or drawing any masks.

| Classes | Seed | Flat retest ACC | Flat retest UA |
|---|---:|---:|---:|
|4|66|77.283|77.524|
|4|67|77.713|77.703|
|4|68|78.275|78.509|
|4|Mean|77.757|77.912|
|6|66|59.691|58.619|
|6|67|60.934|59.185|
|6|68|60.861|59.543|
|6|Mean|60.495|59.116|

All240 original Flat saved prediction files were read; their recomputed ACC
matches metrics.test[rate].accuracy within1e-8. Per-class confusion counts
yield UA; macro recall explicitly includes all4/6 classes. Machine-readable
values are in NEW_MISS_FLAT_UA_AUDIT.json, with absent Test-best UA as JSON null.
These retest UAs must not be attached to Test-best ACC or used as a paired
selected-epoch comparison against B6.
