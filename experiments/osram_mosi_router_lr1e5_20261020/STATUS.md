# Status: two formal runs started

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. No completed performance result yet.

Observed 2026-10-20T09:22:07.097529+00:00: Flat PID1818452 and Router PID1818483 running on
biggpu physicalGPU2 (GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4),150epochs each.
Fresh seed66 initialization and original full data split; both initial backbone
SHA256 values match the completed same-seed reference. Effective config confirms
constant backbone1e-5, unchanged Router3e-4/3e-5 schedule,wd1e-5,batch32,
gcnet-random-mask-v1 and per-rate Test-best W-F1. Only backbone LR changes.

21targeted local tests passed; compileall and git diff checks passed.
Existing optional torch-sparse/libcusparse warning was nonfatal and not used.
One real three-batch GPU smoke passed: finite/nonzero backbone and Router
updates, full Adam/model/RNG recovery and exact selected-prediction reload.
Peak allocated3092.0576171875MiB. This is correctness evidence, not performance.

Frozen training source d152360fcf6b51daf2d29f8281b578b1c456d297;
archive SHA25631f24c5042e94f23e113403f4ccc58085ca8f35bbc5b2f19fb1a6ae546b5cf18.
Remote root /data2/yb/remote_experiments/osram_mosi_router_lr1e5_20261020;
source source_d152360; output execution_v1. tmux socket osram_mosi_lr1e5,
session mosi_lr1e5_d152360, controller PID1816333.

Queue automatically writes final RESULT.md/METRICS.csv/SUMMARY.json after both
150epoch runs finish and paired checks pass. No old checkpoint/results changed.
No additional seeds or datasets authorized by this bounded LR experiment.
