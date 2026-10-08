# MOSI Nested direct replacement

INTERNAL DIAGNOSTIC ONLY

Seed66, 100 epochs, original cfg84 no-JEPA cyclic random missing 0.0–0.7.
Same per-rate BEST selection, optimizer, data, masks and task head. Old residual
Nested reference mean8 W-F1 80.992147, high 76.077229; Flat 81.068095/76.352251.

Only remove TokenAdapter output additions: adapter receives decoded graph
Local/Base/Gap, rather than original evidence plus decoded graph outputs.
Original Local Skip remains. No OSRAM/query/write changes. All decoder zero
initialization retained, so direct graph outputs initially zero; this is NOT
baseline-equivalent initialization. Graph internal operations remain unchanged.
Same parameters and RNG sequence as residual Nested. No random-init decoder
change is bundled into this ablation.

Server biggpu, physical GPU6 only; source snapshot sealed before training.
Entry: python -m experiments.osram_nested_direct_20261008.dispatch --root
/data2/yb/remote_experiments/osram_nested_direct_20261008 --data-manifest
/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json

Status: implementation complete; launch pending verification.
