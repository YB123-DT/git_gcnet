# MELD Flat optimizer screen

INTERNAL TEST-ORACLE SCREEN; seed66 only.

All eight jobs completed50/50 epochs without failures. Best screened setting:
O6, LR1e-5 / Adam weight_decay2e-4. Relative to the reused same-source seed66
W0 control, eight-rate ACC +0.948276pp, UA +2.802146pp, W-F1 +2.128863pp.
This does not replace the previously accepted three-seed baseline: this screen
has one backbone seed and test-set hyperparameter selection.

Only LR and Adam weight_decay differ. Original Flat1600 + Local Skip; no added Block.
Original GCNet random-mask-v1, ECERC,50epochs,batch32,constant LR,cmerc-repro.
ACC-best corresponding UA/W-F1 use saved selected-epoch predictions. No UA-best mixing.

| Job | LR | weight_decay | ACC | UA | W-F1 | high ACC | high UA |
|---|---:|---:|---:|---:|---:|---:|---:|
| W0 reused | 1e-3 | 1e-5 | 60.838 | 35.633 | 56.789 | 56.718 | 29.025 |
| O0 | 0.0002 | 0.0002 | 60.915 | 35.254 | 56.885 | 56.884 | 29.999 |
| O1 | 0.0002 | 0.001 | 60.862 | 35.397 | 56.981 | 56.437 | 29.327 |
| O2 | 0.0001 | 0.0002 | 61.365 | 36.121 | 57.703 | 57.344 | 30.092 |
| O3 | 0.0001 | 0.001 | 61.384 | 35.793 | 57.505 | 57.458 | 29.652 |
| O4 | 3e-05 | 0.0002 | 61.389 | 36.355 | 57.863 | 57.139 | 30.403 |
| O5 | 3e-05 | 0.001 | 61.317 | 36.109 | 57.735 | 57.024 | 30.278 |
| O6 | 1e-05 | 0.0002 | 61.786 | 38.435 | 58.918 | 57.880 | 32.411 |
| O7 | 1e-05 | 0.001 | 61.667 | 37.833 | 58.673 | 57.625 | 32.171 |

Actual initialization and all50 epochs of train/test mask streams are checked against W0.
ACC-selected epochs can differ: matching streams does not imply matching selected masks.
This is one-seed exploratory Test-oracle HPO, not independent paper performance.
No automatic seed expansion. Small LR may require more epochs;50epoch budget is unchanged.
Per-rate metrics, epochs, class recall/support, checkpoint/mask hashes and failures: RESULTS.json.

Post-completion audit on2026-10-18 reloaded all64 selected BEST checkpoints and
their linked saved predictions: hashes, epochs, rates, N2610, mask hashes and
recomputed ACC/UA/W-F1 all matched the summaries. All eight runs retain a full
last_training.pt and50 history entries. No new inference or training was run.
