# Rho0.5 task-only selected epochs

INTERNAL DIAGNOSTIC ONLY. Existing MOSI seed66 A/control,100 completed epochs, two task losses weighted equally; history_drop_prob=.2, no InfoNCE. Not the later rho.1 run or B/contrastive. No new training/inference.

Read results/control/{config,metrics,history}.json; history length100 and selected_epoch_by_rate:

| Rate | BEST epoch |
|---|---:|
|.0|51|
|.1|51|
|.2|68|
|.3|97|
|.4|58|
|.5|51|
|.6|51|
|.7|43|

Median51; range43–97. Final eight-rate W-F1 80.386%, high75.861%. Final selected checkpoint collection obtained by epoch97, driven by miss.3; all other rates selected by68, high-missing rates by51. Not a single global-best epoch. Do not infer all rates need longer training from this one late selection.
