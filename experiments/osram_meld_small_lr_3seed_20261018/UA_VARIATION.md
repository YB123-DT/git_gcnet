# Why does the MELD UA change differently across seeds?

INTERNAL DIAGNOSTIC ONLY. Offline analysis of existing artifacts/history;
no new inference or training. Read alongside UA_VARIATION_ANALYSIS.json.

## Direct numerical explanation: per-class recall changes

ACC = sum_c (N_c/N) Recall_c, whereas UA = (1/7) sum_c Recall_c.
All8 evaluated rates have identical class support, old and new:
`[1256,281,50,208,402,68,345]`, N2610. The following recalls are averages across
the8 rate-specific ACC-selected evaluations, not one checkpoint's deployment.

For seed68:

| Class ID | Support | Old mean recall | New mean recall | Change pp |
|---|---:|---:|---:|---:|
|0|1256|87.371|86.963|−0.408|
|1|281|38.434|42.126|+3.692|
|2|50|14.000|4.250|−9.750|
|3|208|22.776|20.132|−2.644|
|4|402|44.248|46.766|+2.519|
|5|68|15.441|9.007|−6.434|
|6|345|40.217|44.493|+4.275|

Weight these changes by class support: ACC +0.589080pp. Weight them equally:
UA −1.249987pp. Therefore the opposite ACC/UA changes are arithmetically
consistent: declines on classes2/5 matter much more to UA than their small
population shares matter to ACC. It is NOT accurate to attribute this seed's
change simply to improved majority recall: class0 recall also falls.

The pattern differs by seed. Seed66 improves classes1–6 while class0 falls;
seed67 improves classes1/3/4/5/6 but class2 falls by4pp. The corresponding
UA changes against the exact historical artifacts are +2.998953, +1.160706,
−1.249987pp. All classes and rates, including negative changes, remain in JSON.

## What is not yet established

This explains the **score composition**, not the unique optimizer root cause.
Random seed changes initialization, batch order and missing masks. Original
GCNet evaluation also draws fresh masks and each run picks its own per-rate
ACC-best epoch. The historical and new selected masks are not identical pairs.
ACC selection does not maximize UA, even though UA is correctly computed from
the same selected predictions. There is no metric/epoch mixing here.

Both learning_rate and weight_decay changed (.001/1e-5 ->1e-5/2e-4), so these
results cannot isolate learning_rate as the sole cause. Seed66 also selected
this configuration out of8 tested settings, so its largest improvement must not
be assumed representative of unseen seeds.

End-of-training histories demonstrate different optimization trajectories:

| Seed | Old epoch50 train CE | New epoch50 train CE | Old train ACC | New train ACC |
|---|---:|---:|---:|---:|
|66|0.674|0.900|76.234|68.475|
|67|0.645|0.902|77.165|68.916|
|68|0.672|0.918|76.594|68.315|

These rounded values describe epoch50 training, not the selected test scores.
The new model fits the training data less strongly at this budget, consistent
with changed optimization/regularization. This does not prove undertraining,
overfitting, a particular Memory defect or that longer training would help.
Seed68's selected new epochs span4–41, so “all rates simply need more epochs”
is not established. Saved snapshots and exact per-rate epochs are in JSON.

Conclusion: UA improvement is seed-sensitive and specifically loses minority
class recall on seed68. The three-seed mean improvement is real under the
recorded exploratory protocol but is not uniform class/seed improvement.
