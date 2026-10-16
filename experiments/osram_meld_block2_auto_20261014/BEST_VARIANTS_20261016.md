# Verified MELD variant extrema, 2026-10-16

INTERNAL TEST-ORACLE SCREEN. No new training or inference.

Scope: current corrected `gcnet-random-mask-v1`, eight-rate mean, ACC-selected
per-rate checkpoint and same-selected-prediction UA. Excludes old Nested
conversation-mask results, frozen intervention scores, and common-mask replay
from this Test-best ranking. Different search sources/hyperparameters remain
exploratory observations, not a strictly paired cross-study comparison.

Remote scan inspected98 eligible summary entries (including duplicate reports)
under `/home/qinjinghui/remote_experiments` and
`/data2/yb/remote_experiments`. Config protocol and `testbest.mean` were read;
entries were deduplicated by metrics/method/seed when ranking. Final source
tables were also checked. No prediction regeneration was performed.

| Criterion/configuration | Seed | ACC (%) | Corresponding UA (%) | Evidence |
|---|---:|---:|---:|---|
| Highest ACC: B6 rank2/steps14, threshold0.1, multiplier1 |66|61.216475|35.935447|`../osram_meld_b6_structure_hpo_20261015/RESULT.md`, trial3|
| Tied highest ACC: B6 rank3/steps8 |66|61.216475|34.595023|same report, trial11|
| B6 rank2/steps14, threshold0.33599569865951234, multiplier2.8200380294401013 |66|61.192529|36.007932|remote `osram_meld_b6_threshold_hpo_20261015/RESULT.md`, trial11|
| Highest UA: B6 rank2/steps14, threshold0.1, multiplier0.1 |66|60.636973|38.099011|same threshold report, trial3|
| B6 rank1/steps24 |66|60.933908|37.420921|structural report, trial2|
| Highest B3 ACC: tokens1/rounds3, default local capacity |66|61.020115|35.472568|remote `osram_meld_b3_gpu3_extension_20261016/RESULT.md`, add03|
| B3 default local capacity, tokens4/rounds3 |66|60.756705|37.357830|`../osram_meld_b3_structure_hpo_20261016/RESULT.md`, trial4|

Among the verified completed current-protocol three-seed variant reports,
B6 rank2/steps14 default threshold/LR has mean ACC60.827 (sample SD0.345),
UA35.518 (SD0.790),W-F1 56.801 (SD0.394). The threshold/LR-tuned variant has
ACC60.809/UA36.192. Source:
`../osram_meld_b6_top2_matched_flat_3seed_20261015/RESULT.md`.
The matched Flat is ACC60.635/UA36.304; the accepted historical Flat reference
remains ACC60.693/UA36.255 and is not replaced. The highest single-seed ACC
must not be reported as a three-seed average.

The legacy Nested table contains seed66ACC61.255 and three-seed mean61.153,
but belongs to the earlier missing-protocol screen; it is not eligible for
the current corrected-protocol ranking. None of these selected extrema
establishes a successful second block or robust joint ACC/UA improvement.

## Highest-ACC B6 training hyperparameters verified

Structural trial3 (rank2/steps14) retains previously searched trial16 training
settings,not original Flat optimizer/dropout defaults. Its actual global LR is
0.0003433234536467373,weight decay0.0000044857894231052665,and dropout
0.22489055023388654. Original Flat defaults are LR0.001,weight decay0.00001,
and dropout0.5. The B6 structure-control LR multiplier is1,not a separate
lower adapter LR. Thus this run's global LR is about0.3433times original.

The completed three-seed matched comparison holds these searched global
settings fixed: B6 ACC60.826948/UA35.517778 versus Flat ACC60.635377/UA36.303932,
or +0.191571ACC points and -0.786154UA points. The higher selected ACC does
not establish better seven-class recall. This is a record-only clarification;
no new run or inference was launched.
