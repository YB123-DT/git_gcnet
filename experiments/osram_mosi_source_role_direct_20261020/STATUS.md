# MOSI direct-source run launched

INTERNAL TEST-ORACLE SCREEN

2026-10-20 12:39 UTC: formal training running, epoch1/150 verified.
This is a launch record, not a completed performance result.

- Immutable source: bd2e12ec22765cc10f98dfc38474fed4d48c4e96.
- Server: biggpu; physical GPU0, UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
- Controller PID2585526; formal worker PID2588018.
- Persistent tmux socket osram_source_role_direct, session mosi_direct_bd2e12e.
- Output: /data2/yb/remote_experiments/osram_mosi_source_role_direct_20261020/execution_v1.
- Runtime: original s0 Python, Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2.

One new MOSI seed66 run only:150epochs,Flat1600,8heads,64-d K/V,
write_step.6,Adam,batch32,original scalar MSE and GCNet random-mask-v1.
All original parameters train jointly with the96,487 added parameters.
LR remains.001 for100epochs, then.0001 for50; per-rate earliest Test-best
W-F1 retains matching ACC/predictions/exact masks. No extra datasets/seeds/HPO.
Local/Local Skip/CHM/query do not change. Existing runs are untouched.

Direct output is sum_s 2*sigmoid(score)*source_read, without original read
add-back. Initial active weights all equal1. Initial smoke predictions happened
to match exactly (max error0); direct re-summation is only guaranteed within
the declared tolerance, not bitwise in general. Original frozen backward slots
remain zero. This changes the effective coefficient range as well as removing
the explicit add-back expression; do not call it a range-controlled ablation.

Real MOSI smoke passed:2 actual batches/optimizer steps/Memory scans;
29.805s;peak3093.751MiB. Original backbone and new-module parameters updated;
complete optimizer/model/RNG recovery and selected-prediction reload passed.
Initial source reconstruction maximum errors:Memory4.77e-7,prewrite5.36e-7,
Base3.58e-7,Gap2.98e-7. No performance claim follows from this smoke.

First formal epoch matched the completed Flat training mask hash, LR and
optimizer update count exactly. New-module gradient hooks connected14tensors,
with2nonzero in the first epoch. Zero final layers can initially block upstream
gradients; connectivity is not a claim that all14tensors have already received
nonzero updates. Later epoch logs retain actual coverage.

The controller will finish only this run and then automatically recompute
selected metrics and compare matched Flat/residual/direct results. Eight
rate-specific BEST weights and full last.pt are retained remotely. Different
rates may select different epochs; these are not one-checkpoint deployment
scores. LAUNCH.json records config, source/data/mask hashes, environment,
reference provenance and actual process identity. No weights are uploaded.
