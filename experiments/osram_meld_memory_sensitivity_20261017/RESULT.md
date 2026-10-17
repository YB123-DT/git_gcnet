# Learned MELD Flat Memory parameters

INTERNAL DIAGNOSTIC ONLY. Completed read-only CPU extraction on biggpu at
2026-10-17T10:45:02Z. No GPU,model inference,backward,optimizer step or retraining.

## Source identity and scope

Use only the accepted original ECERC Flat1600/Local-Skip baseline,not B3/B6,
no-Skip,alternate-feature or legacy-mask weights. For seeds66/67/68,loaded
eight ACC-selected BESTs each,24checkpoint states total. These are three
cyclic-trained models at selected epochs,not24independent training runs or
final-epoch weights. Checkpoint SHA256,selected epoch and stored ACC all match
the archived baseline records for all24files.

Pinned baseline mathematical source:b4a313841f098487a6482d2712bdac94aa8e319e.
Config verified in every checkpoint:MELD,correct seed,gcnet-random-mask-v1,
Flat readout,no meaningful Block,Local Skip retained,write_step0.6,
beta_mode=embedded,write_ridge0.001. Source roots are the protected paths in
experiments/osram_meld_testbest_ua_20261014/BASELINE.md. Full paths/hashes/config
and per-checkpoint/head data are in ALPHA_BETA.json.

## Actual trained values

Means pool the eight selected checkpoints and their heads within each seed;
beta also pools A/T/V. Ranges include all those selected parameter entries.
They describe selected states,not independent-sample uncertainty estimates.

| Seed | Alpha mean | Alpha min–max | Beta mean | Beta min–max |
|---:|---:|---:|---:|---:|
|66|0.974787|0.961442–0.979076|0.499392|0.494956–0.502875|
|67|0.975972|0.967881–0.979472|0.499456|0.494073–0.503763|
|68|0.974309|0.968558–0.978365|0.499266|0.491824–0.503255|

For context only,source constructor initialization is alpha0.98,beta0.5.
All192selected alpha entries are below0.98. Maximum absolute changes from
initialization across selected entries are0.018558for alpha and0.008176for beta.
Thus these are learned,changed values; reporting only defaults would be wrong.

| Seed | Mean Beta-A | Mean Beta-T | Mean Beta-V |
|---:|---:|---:|---:|
|66|0.499804|0.498721|0.499649|
|67|0.499748|0.498922|0.499699|
|68|0.499768|0.498368|0.499663|

This table does not measure modality usefulness. Embedded beta acts through
the coupled ridge solve,and cannot be read as a direct content-importance gate.

## Saturation and interpretation

- Across selected alpha values,sigmoid derivative minimum is0.020107. Since
  every alpha is below0.98 and above0.96,the derivative is larger than the
  initialization derivative0.0196. These trained values are not more
  sigmoid-saturated than initialization; this does not prove optimal retention.
- Across selected beta values,sigmoid derivative minimum is0.249933,close to
  the mathematical maximum0.25. Beta sigmoid saturation is not supported.
- Alpha lies0.961442–0.979472:there is no selected head with near-zero retention
  or complete forgetting each step. Later writes may still interfere with old
  associations; alpha alone does not measure realized history survival.
- Beta stays close to0.5,but that does not prove a disconnected gradient or an
  ineffective writer. See REVIEW.md for embedded-beta/ridge algebra. Its small
  parameter movement cannot isolate task sensitivity,optimization or weight
  decay effects.
- Parameter values cannot establish which head is useful or dead. No head
  ablation,read-gradient measurement or new functional intervention was run.

## Selected epochs

Columns are missing rates0.0through0.7.

| Seed | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|66|11|11|5|21|5|34|24|10|
|67|6|12|4|12|12|7|18|24|
|68|17|20|17|14|10|10|7|19|

ALPHA_BETA_PER_HEAD.csv gives192seed/rate/head rows with alpha,Beta-A/T/V
and sigmoid derivatives. ALPHA_BETA.json additionally retains logits and
retention-only alpha^8 values. EXTRACTION_COMMAND.txt reproduces the read-only
extraction;it never writes to the protected remote roots.

Conclusion:the specific explanation "trained retention collapses toward0or
beta saturates toward0/1" is not supported by these accepted BEST weights.
This is not a diagnosis of Query/Decision optimality,nor a claim that Memory
hyperparameter changes cannot help. No sensitivity sweep or training was started.
