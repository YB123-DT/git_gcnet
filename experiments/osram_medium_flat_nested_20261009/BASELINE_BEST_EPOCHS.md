# Original large Flat seed66: when the per-rate best was reached

INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle. Source: biggpu `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/{metrics,history}.json`. All selected epochs and W-F1 values checked against earliest maxima in the full100epoch history. No training or inference performed.

| Missing rate | Selected epoch (1-based) | W-F1 (%) |
|---|---:|---:|
|.0|76|88.205138|
|.1|76|86.506580|
|.2|76|83.186756|
|.3|76|80.762955|
|.4|76|80.826582|
|.5|48|77.494019|
|.6|76|75.789556|
|.7|39|75.773178|

The final mean8 81.068095% is first attained by the accumulated per-rate-best collection at epoch76; it is NOT a single epoch76 checkpoint score across eight rates. No rate improved its selected W-F1 after epoch76 through100.

| Training completed through epoch | Accumulated per-rate-best mean8 W-F1 (%) |
|---|---:|
|40|79.922886|
|50|80.361308|
|60|80.464312|
|76|81.068095|
|100|81.068095|

This establishes that the original model improved late; it does not guarantee that the new adapter widths will follow the same trajectory. Compare final equal budgets before drawing width conclusions.
