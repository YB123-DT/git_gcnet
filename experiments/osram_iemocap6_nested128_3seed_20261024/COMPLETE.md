# Complete — IEMOCAP6 Nested128 five folds, three seeds

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

Verified2026-10-24: queue complete/error null,14 new complete100-epoch workers
plus the preserved seed66/fold5 result. Strict remote report accepted all15
trajectories. Downloaded only small RESULT.md/JSON and per-fold/per-seed CSVs
to completed/; their SHA256 digests match the remote generated files. Recomputed
three-seed aggregate arithmetic and checked120fold/rate +24seed/rate records.
No new training, checkpoint deletion, test inference or selection was performed
during this completion audit. All weights, masks and predictions stay remote.

Main source45d06f2ec780c450e9f1a6dceaaca072cbd2d76b; reused fold source3777ab1.
One diagnosed GPU OOM was explicitly recovered from complete epoch11 with Adam
and RNG using controller-only recovery; see RECOVERY.md. Its failed-run evidence
remains retained. Recovery did not change model,mask,loss,batch or optimizer.

## Reference comparison with historical Flat

First equally average8rates per fold, then5folds per seed, then3seeds.

|Seed|Flat ACC|Nested ACC|Delta pp|Flat UA|Nested UA|Delta pp|
|---|---:|---:|---:|---:|---:|---:|
|66|62.011|62.009|-0.003|59.945|60.574|+0.630|
|67|61.624|61.792|+0.168|60.052|60.025|-0.027|
|68|61.507|62.180|+0.673|60.307|60.807|+0.500|
|Mean|61.714|61.994|+0.280|60.101|60.469|+0.367|

Nested three-seed sampleSD: ACC0.195,UA0.402,WF10.264. MeanWF161.608 versus
historical Flat61.362 (descriptive difference+0.246pp). These small differences
do not establish a stable/significant module benefit.

Historical Flat sources:
`/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016`
and `/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016`.
Re-read all15Flat fold summaries and recomputed the table rather than mixing
oldMissing/s0/4class/single-fold results. Both are cmerc-repro,newGCNetmissing,
100epochs,LR1e-3,WD1e-5,batch32,per-rate TEST ACC. Feature roots and label hash
match; the sampler source SHA matches. However historical Flat resampled test
masks per epoch; new Nested replays fixed accepted Flat66 masks for every seed.
The fold-initialization treatment/source snapshot also differ. Therefore this
is a historical score comparison, NOT a strictly paired Nested ablation.
Do not interpret the table as isolated causal module gains.

Final remote output:
`/data2/yb/remote_experiments/osram_iemocap6_nested128_3seed_20261024/execution_v1`.
No further seed/fold/HPO expansion is authorized by completion.
