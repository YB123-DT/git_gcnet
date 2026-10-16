# Existing B6-versus-Flat comparisons, separated by seed

INTERNAL TEST-ORACLE DIAGNOSTIC. Interpretation of this request: separately
trained Flat without B6 versus Flat with B6, NOT inference-time Local/Base/Gap
zeroing and NOT switching off B6 inside the same trained checkpoint. No new
training, inference, checkpoint changes or experiment launches.

## MELD: all seeds

50epochs, original GCNet missing, eight-rate average, ACC-selected exact-epoch
ACC/UA pairs. Matched Flat uses the SAME global hyperparameters as B6:
lr0.0003433234536467373, weight_decay0.0000044857894231052665,
dropout0.22489055023388654, batch32. It is a module-ablation control, not a
replacement of the accepted original-hyperparameter Flat60.693 baseline.

| Seed | Matched Flat ACC | B6 tuned ACC | Difference(pp) | Matched Flat UA | B6 tuned UA | Difference(pp) |
|---|---:|---:|---:|---:|---:|---:|
|66|60.781|61.193|+0.412|37.107|36.008|−1.099|
|67|60.321|60.445|+0.125|36.161|37.193|+1.033|
|68|60.805|60.790|−0.014|35.645|35.376|−0.269|
|Mean|60.635|60.809|+0.174|36.304|36.192|−0.112|

For completeness, the other already-completed B6 rank2/14-step configuration
uses default threshold.1 and B6 LR multiplier1 rather than tuned threshold
0.33599569865951234 and multiplier2.8200380294401013:

| Seed | B6 default ACC | B6 default UA |
|---|---:|---:|
|66|61.216|35.935|
|67|60.560|36.011|
|68|60.704|34.606|
|Mean|60.827|35.518|

All seeds are reported; tuned seed67 improves both ACC and UA, seed66 improves
ACC while reducing UA, seed68 reduces both. Do not present seed66's61.2ACC
as evidence of joint ACC/UA improvement. These runs use the same sampling law
but are NOT shared-exact-mask paired scores; auxiliary common-mask replay
for67/68 remains incomplete, as documented in the original matched report.

Source: ../osram_meld_b6_top2_matched_flat_3seed_20261015/{SUMMARY.json,RESULT.md}.
Rechecked all nine configurations/seeds and72 rate summaries; per-rate means
agree with their stored summaries within1e-8.

## IEMOCAP: seed66 chosen before inspecting this comparison

100epochs, original GCNet missing, equal fivefold/eight-rate ACC means.
The global original IEMOCAP lr.001, dropout.5, weight_decay1e-5, batch32 are
unchanged; ONLY the tuned B6 settings were transferred from MELD. Thus the
MELD and IEMOCAP global training hyperparameters are deliberately different.

| Task | Flat ACC | B6 tuned ACC | Difference(pp) |
|---|---:|---:|---:|
| IEMOCAP4, seed66 |79.198|79.408|+0.210|
| IEMOCAP6, seed66 |61.175|61.637|+0.462|

Original Flat source: corrected-mask20261013_r1 metrics.selected_accuracy_by_rate;
B6 source: this experiment SUMMARY.json, fivefold/seeds/66/mean. Cross-checked
against MISSING_PROTOCOL_SEED_CHECK.json. These are not fold5-only scores.
No original Flat exact-selection UA is claimed. The complete three-seed ACC
differences remain only+0.026pp/+0.109pp; the chosen seed cannot replace them
or establish a robust multi-seed method benefit.

The older Local/Base/Gap source-combination diagnostic is a different
intervention and, for some retained runs, a different missing protocol. Its
numbers are intentionally not relabelled as these B6 module ablation results.
