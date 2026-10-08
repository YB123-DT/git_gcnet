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
Status: complete; final results below supersede the launch status.


## Completed seed66 result

100/100 epochs completed, exit0, outputs_verified=true. Saved config, metrics,
history, recovery checkpoint, eight BEST checkpoints and eight prediction files
verified against recorded SHA256. No new training or inference for this report.

|Model|Mean8 ACC|Mean8 W-F1|High ACC|High W-F1|
|---|---:|---:|---:|---:|
|flat|81.174|81.068|76.524|76.352|
|old_nested|81.059|80.992|76.169|76.077|
|direct|77.439|77.315|71.392|71.312|

|Rate|Residual ACC|Direct ACC|Residual W-F1|Direct W-F1|
|---|---:|---:|---:|---:|
|0.0|88.110|87.043|88.078|86.943|
|0.1|86.433|83.994|86.358|84.074|
|0.2|83.841|80.335|83.735|79.681|
|0.3|80.488|78.963|80.523|79.051|
|0.4|81.098|75.000|81.012|74.836|
|0.5|77.896|70.884|77.675|70.964|
|0.6|75.152|71.037|75.032|71.100|
|0.7|75.457|72.256|75.525|71.872|

Direct minus residual: mean8 W-F1 -3.677139pp; high W-F1 -4.765380pp.
All eight rates decrease in W-F1. Retain residual design for now; no automatic
multi-seed expansion. This supports the tested residual configuration, not a
universal claim that all direct replacements fail: zero decoder initialization
was intentionally shared and direct starts with zero adapter evidence.
