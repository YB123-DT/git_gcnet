# Incomplete three-seed IEMOCAP snapshot

INTERNAL DIAGNOSTIC ONLY. Not a full five-fold aggregate.
Queue at inspection:20complete,4running,3pending; no reported error.
Metrics are per-rate Accuracy-selected BEST checkpoint results, not the
latest training epoch. Rates0.0–0.7 and included folds equally weighted.
UA recomputed using classification_scores from acc_ua.py and existing saved
predictions; ACC matches metrics.json within1e-12. No new inference/training.
Only include completed provenance-verified folds. No missing-fold imputation.

|Dataset|Completed subset used|Seed|W-F1 (%)|ACC (%)|UA (%)|
|---|---|---:|---:|---:|---:|
|Four|fold1–4|66|78.414548|78.468695|78.843544|
|Four|fold1–4|67|77.542485|77.732769|78.508370|
|Four|fold1–4|68|78.109541|78.206197|79.074978|
|Six|fold1–3|66|60.180863|60.819706|60.471750|
|Six|fold1–3|67|60.541258|61.031964|60.265676|
|Six|fold1–3|68|60.741867|61.298113|60.910587|

Six seed66 already has all five completed folds (full mean W-F160.431700,
ACC60.878719,UA59.402902). Here its first three folds are used ONLY to show
the same completed subset as seeds67/68; do not confuse this partial score
with the previously reported full fivefold score.

Remaining at inspection: Four seed66fold5 epoch25, seed67fold5 epoch7;
Six seed67fold4 epoch73,seed68fold4 epoch49 running. Four seed68fold5 and
Six seed67/68fold5 pending. Wait for all before claiming full three-seed
performance; do not compare these partial subsets with historical full folds.
