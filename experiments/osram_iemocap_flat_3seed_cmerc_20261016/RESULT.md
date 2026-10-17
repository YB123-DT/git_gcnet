# Original Flat three-seed completion

INTERNAL PER-RATE TEST-ORACLE ACC; original Flat seeds66/67/68; cmerc-repro

Complete fold runs: 30/30 (10 reused; 20/20 new complete).

| Dataset | Fivefold three-seed scope | ACC | UA | W-F1 |
|---|---|---:|---:|---:|
| IEMOCAPFour | mean | 79.351 ± 0.242 | 79.632 ± 0.368 | 79.233 ± 0.253 |
| IEMOCAPFour | high | 76.884 ± 0.361 | 77.278 ± 0.528 | 76.725 ± 0.387 |
| IEMOCAPSix | mean | 61.714 ± 0.264 | 60.101 ± 0.186 | 61.362 ± 0.065 |
| IEMOCAPSix | high | 59.277 ± 0.380 | 57.969 ± 0.140 | 58.843 ± 0.054 |

## Per-seed eight-rate fivefold means

| Dataset | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| IEMOCAPFour |66|79.549|80.024|79.439|
| IEMOCAPFour |67|79.081|79.293|78.951|
| IEMOCAPFour |68|79.423|79.579|79.308|
| IEMOCAPSix |66|62.011|59.945|61.382|
| IEMOCAPSix |67|61.624|60.052|61.415|
| IEMOCAPSix |68|61.507|60.307|61.289|

Protocol:100epochs,original Flat1600/Local Skip,no Block2,GCNet random-mask-v1,
cyclic0.0–0.7;per-rate Test-best ACC. UA uses the same ACC-selected predictions,
not a separate UA-selected epoch. First average8rates within eachfold,then
weight5foldsequally,then3seedsequally. This is not pooled-utterance ACC/UA.
Runtime is cmerc-repro,not the historical adapter environment. Numerical source:
785a01242ef1a82b97d5337acbb33ba963df5537.

Remote verification2026-10-17:30completefoldsummaries,stored per-class recall
UA/rate/fold/seed aggregates agree;240BEST checkpoints and30full recovery states
exist. See STATUS.md. No retraining or fresh randomized inference was performed.

Fivefold means require all five folds; three-seed mean ± sample SD appears only after all30 folds complete.
Seed66 outputs are reused read-only. New training is limited to Flat67/68; all checkpoints/artifacts/full recovery states are retained.
Completed B6 reference: /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015. No B6 run is submitted. Test-best results are internal per-rate test-oracle ACC diagnostics.
