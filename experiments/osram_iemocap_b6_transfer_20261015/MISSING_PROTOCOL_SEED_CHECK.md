# Historical missing-protocol comparison, by seed

INTERNAL TEST-ORACLE DIAGNOSTIC. No new training or inference.

All runs are100epochs and fivefold/eight-rate means. Old Flat source:
`/data2/yb/remote_experiments/osram_iemocap_cfg84_nojepa_5session_20260919`.
Corrected GCNet Flat and current B6 source paths are preserved in the JSON.

## Selection-rule mismatch in the original old records

Old metrics omit explicit selection_metric and selected_accuracy_by_rate.
Inspecting all240 stored selected epochs against their100-epoch histories
shows240/240 attain maximal W-F1, but47/240 do not attain maximal ACC (193
attain both). Therefore old stored ACC must not silently be labelled ACC-best.

For the following old-Flat column ONLY, select the first maximal-ACC epoch
per rate from the already recorded100-epoch history. This is an explicitly
labelled offline analysis, not retraining, replay inference, or replacement
of historical records/checkpoints. JSON preserves old stored values too.

| Classes | Seed | Old Flat, offline ACC-best | GCNet Flat, ACC-best | B6, ACC-best | B6−GCNet Flat(pp) |
|---|---:|---:|---:|---:|---:|
|4|66|79.291445|79.197593|79.407509|+0.209916|
|4|67|79.274229|79.096995|79.154762|+0.057767|
|4|68|79.179509|79.487403|79.298058|−0.189345|
|4|Mean|79.248394|79.260664|79.286776|+0.026113|
|6|66|61.411025|61.174863|61.637348|+0.462485|
|6|67|61.482534|62.046264|61.745636|−0.300628|
|6|68|61.072619|61.653327|61.818374|+0.165047|
|6|Mean|61.322059|61.624818|61.733786|+0.108968|

The originally stored old WF1-selected ACC means remain79.213965/61.220573;
do not overwrite or relabel them with the offline ACC-best analysis above.

Descriptive GCNet-minus-old-ACC-best differences are+0.012269pp (4-class),
+0.302759pp (6-class). Individual6-class seeds67/68 differ by+0.563730/
+0.580708pp, while seed66 differs by−0.236162pp. Similar means do not prove
mask protocols have no effect. The historical runs are not an isolated,
same-source/shared-exact-mask missing-algorithm ablation; selection-time
mask randomness also differs. Do not attribute every difference solely to
changing the sampler. B6's separate comparison uses the corrected GCNet
Flat throughout, not any old-protocol score.
