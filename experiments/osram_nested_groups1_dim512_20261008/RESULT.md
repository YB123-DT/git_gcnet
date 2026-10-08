# MOSI: one 512d node per memory evidence

INTERNAL DIAGNOSTIC ONLY

Seed66, 100 epochs; old residual Nested, not direct replacement. Preserve
OSRAM eight heads x64, read/write/query, Flat, original Local Skip, original
task loss, cyclic random missing, per-rate BEST and training configuration.

New variant nested_groups1_dim512 uses one 512d graph node per Base/active
Gap, instead of eight64d nodes. Input/output memory slots remain512d.
Same three-layer rooted GIN; zero output decoders and residual additions.
The existing shared-width graph requires Local to project256->512 as well
(old graph Local256->64); original external Local and Skip remain256d.
Thus memory per-evidence graph width is matched (1x512=8x64), but Local graph
width, parameter count and topology are NOT matched. Do not call this a pure
head-count-only ablation or parameter-matched experiment.

References seed66 W-F1 mean8/high: original8x64 80.992/76.077;
prior1x64 80.784/76.236. Final reporting uses both ACC and W-F1.

Run root /data2/yb/remote_experiments/osram_nested_groups1_dim512_20261008.
Server biggpu GPU6, UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
Command: python -m experiments.osram_nested_groups1_dim512_20261008.dispatch
--root RUN_ROOT --data-manifest
/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json

Measured module parameters: original8x64=159235; prior1x64=158339;
new1x512=4472579. Corresponding full model count expected17982372.
CPU checks passed: grouping/interface dimensions, zero-initialized identity,
inactive/padding/NaN masks, finite gradients and actual core update.
Trainer import checked in the sealed complete source archive.

Status: running. Launched2026-10-08 10:58 UTC, snapshot2aac302;
tmux nested_groups1_dim512_20261008, dispatcher PID3369959, training PID3370206.
GPU6 had23779MiB free at admission. Existing random-decoder experiment retained.
Live DISPATCH.json/seed_66/train.log record progress; no completed score yet.
