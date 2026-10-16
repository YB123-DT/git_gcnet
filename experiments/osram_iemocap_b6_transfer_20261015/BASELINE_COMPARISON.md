# Original Flat comparison: distinguish selected scores from fresh-mask retesting

INTERNAL TEST-ORACLE DIAGNOSTIC. Recomputed2026-10-16 from both methods'
original per-fold metrics/config/history files; no training or inference.

## Source evidence and checks

Original Flat:
`/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/iemocap{4,6}/seed_{66,67,68}/fold_{1..5}/`.

B6:
`/home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015/runs/iemocap{4,6}/seed_{66,67,68}/fold_{1..5}/`.

Read `metrics.json`, `config.json` and `history.json` in all60 directories.
Both methods have100 completed epochs, GCNet-random-mask-v1 and ACC selection.
All original configuration keys remain identical except logical device spelling
and `osram_meaningful_block`. New B6-specific settings and explicit artifact
bookkeeping are documented separately in README.md; MELD global LR/dropout/WD
were not copied. Every B6 selected accuracy in metrics.json matches its already
verified exact-selection SUMMARY.json result.

For Test-best, read `selected_accuracy_by_rate` and
`selected_weighted_f1_by_rate`, NOT `test[rate]`. Average eight rates per fold,
five folds per seed, then three seeds equally. The `test` dictionary contains
a later fresh stochastic-mask evaluation and is a separate measurement.

## Same selection rule: original Flat versus B6

All values percentages; differences percentage points.

| Dataset | Seed | Flat ACC | B6 ACC | Difference | Flat W-F1 | B6 W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP4 |66|79.197593|79.407509|+0.209916|79.057098|79.263454|
| IEMOCAP4 |67|79.096995|79.154762|+0.057767|79.007924|79.033602|
| IEMOCAP4 |68|79.487403|79.298058|−0.189345|79.411910|79.184315|
| IEMOCAP4 |Mean|79.260664|79.286776|+0.026113|79.158977|79.160457|
| IEMOCAP6 |66|61.174863|61.637348|+0.462485|60.520228|61.239426|
| IEMOCAP6 |67|62.046264|61.745636|−0.300628|61.398533|61.462224|
| IEMOCAP6 |68|61.653327|61.818374|+0.165047|61.189239|61.349663|
| IEMOCAP6 |Mean|61.624818|61.733786|+0.108968|61.036000|61.350438|

W-F1 changes: IEMOCAP4 +0.001480pp, IEMOCAP6 +0.314438pp.
ACC does not improve consistently in every seed. These are stochastic
per-rate-Test-best comparisons, NOT shared-exact-mask paired evaluation or
evidence of statistically significant gains. Original exact-selection UA is
unavailable and is not fabricated from final retest predictions.

## Why the remembered77/60 cannot be the Test-best baseline

Reading `metrics.json:test[rate].accuracy` instead gives:

| Dataset | Flat fresh-mask retest ACC | B6 fresh-mask retest ACC | Difference |
|---|---:|---:|---:|
| IEMOCAP4 |77.756928|77.680610|−0.076318|
| IEMOCAP6 |60.495308|60.433349|−0.061959|

Both Test-best and final-retest values exist in the same files. Mixing Flat's
77.757/60.495 retest values with B6's79.287/61.734 selected values would create
a spurious large apparent gain. B6's previously reported selected scores were
correct; reporting them without a verified baseline alongside made this
comparison ambiguity easier to miss. There is no large ACC improvement here.
