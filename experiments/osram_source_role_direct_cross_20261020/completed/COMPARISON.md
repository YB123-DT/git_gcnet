# Completed direct-source weighting: comparison boundaries

INTERNAL TEST-ORACLE SCREEN

All21approved runs completed: MOSEI three seeds60epochs,MELD three seeds50epochs,
IEMOCAP-6 fivefold×three seeds100epochs. No failed or partial run enters means.
The change is gamma=2*sigmoid(score) direct source sums; original Base/Gap are
not added back. LocalSkip,Flat1600 and original causal CHM remain.

## Three-seed results

Eight-rate equal means; IEMOCAP first averages five folds within seed/rate.
Percentages; ± is sampleSD across three seed means,not across rates or folds.

| Dataset | ACC | UA | W-F1 | high ACC | high UA | high W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MOSEI |83.453±0.160|N/A|83.261±0.169|80.398±0.174|N/A|80.074±0.170|
| MELD |61.569±0.138|37.058±0.334|58.319±0.289|57.258±0.092|31.476±0.760|52.682±0.571|
| IEMOCAP-6 fivefold |60.729±0.100|58.882±0.092|60.360±0.046|57.858±0.374|55.825±0.387|57.388±0.232|

## Existing Flat references, not an invented new control

| Dataset/scope | Reference ACC | Direct ACC | delta pp | Reference UA | Direct UA | delta pp |
|---|---:|---:|---:|---:|---:|---:|
| MELD three-seed mean |61.557|61.569|+0.013|37.728|37.058|-0.670|
| IEMOCAP-6 three-seed fivefold mean |60.708|60.729|+0.020|58.706|58.882|+0.176|

MELD reference is the accepted registry
`../../osram_meld_wd_seed66_20261018/BASELINE.json`. Direct W-F1 is lower by0.189pp.
Both use original GCNet mask sampling and per-rate ACC-best same-call UA,
but different trained trajectories/selected epochs can have different exact
test masks. The new driver's explicit training-RNG reset is not proven identical
to the historical source continuation. Therefore these are reference score
differences,NOT same-checkpoint/same-mask interventions or strict trajectory
pairing. Do not call the small ACC difference a meaningful improvement.

IEMOCAP reference is the completed full-training-fold suite
`../../osram_iemocap6_gcnet_testbest_20261020/METRICS.csv`,Flat rows only.
Same100epoch budget,GCNet-v1 and fold-specific fixed legal evaluation banks;
all five folds and all three seeds included. Mean W-F1 changes by+0.045pp.
The three seed-level changes are not uniform:

| Seed | Flat ACC | Direct ACC | delta pp | Flat UA | Direct UA | delta pp |
|---|---:|---:|---:|---:|---:|---:|
|66|60.497|60.803|+0.306|58.659|58.945|+0.286|
|67|61.447|60.769|-0.679|59.500|58.925|-0.575|
|68|60.180|60.615|+0.434|57.959|58.777|+0.817|

MOSEI currently has only the completed seed66 matched Flat reference at
`/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat`.
Compare seed66 only: ACC83.492708→83.392955(-0.099752pp),
W-F1 83.279076→83.243611(-0.035465pp); high W-F1 80.181722→79.980188(-0.201534pp).
Do not compare the new three-seed mean to a one-seed reference as if both were
matched three-seed results. No extra Flat training was authorized or launched.

## Interpretation

This direct weighting has NOT shown a clear,consistent overall gain in this
finite suite. MELD's nearly unchanged ACC accompanies lower UA; IEMOCAP's mean
gain is small and changes direction between seeds; MOSEIseed66 is near its
Flat reference. These observations do not establish statistical significance,
identify a causal failure mechanism,or disprove all source-aware methods.
No additional tuning,training,diagnostic inference or experiments are started.

The result remains per-rate Test-oracle exploration,not independent-test
generalization or a single deployment checkpoint. Full per-rate/per-fold/per-seed
scores and selected epochs are retained in RESULT.md,METRICS.csv,REPORT.json.
