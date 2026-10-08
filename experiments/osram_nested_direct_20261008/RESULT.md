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

CPU verification passed: exact shared parameter initialization and RNG,
output-addition identity, inactive/padding/NaN masking, finite gradients and
actual core parameter update. No additional training smoke run.

Initial launch failed before training: restricted source archive omitted root
config.py (ModuleNotFoundError). Original logs retained. Corrected deployment
uses a full tracked git archive and passed trainer import verification.

Relaunched 2026-10-08 09:55 UTC on biggpu GPU6, UUID
GPU-e4cafb17-818e-216a-b94a-7440063a9153. Sealed code fee9828,
tmux nested_direct_attempt2_20261008, dispatcher PID3071477, training PID3071774.
Remote root: /data2/yb/remote_experiments/osram_nested_direct_20261008/attempt2.
DISPATCH.json tracks terminal status; seed_66/train.log is the training log.
On completion SUMMARY.json compares existing Flat/residual Nested with direct.
Status: running; no completed performance result yet.
