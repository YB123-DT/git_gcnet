# Seed66 WD completion: actual launch

INTERNAL TEST-ORACLE SCREEN. Training running; final scores not yet available.

Code3628d1e69d611f7d6a8a0d6896b435a63f805a9c, pushed to github remote,
YB123-DT/git_gcnet feature/ecerc-features. Immutable snapshot on biggpu:
`/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/source_3628d1e`.

Output `/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1`.
tmux `meld_wd66_1018`; controller PID3228217. Four workers simultaneously running:

| Job | seed | WD | physical GPU | PID |
|---|---:|---:|---:|---:|
|S66_WD0|66|0|2|3228227|
|S66_WD1|66|1e-5|2|3228235|
|S66_WD2|66|1e-4|2|3228239|
|S66_WD3|66|5e-4|2|3228243|

GPU2 UUID GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4, logicalcuda:0. Observed
worker allocations about1.2GiB each; four fit32GB. GPU4 never used.
No other task terminated/moved; GPU0 B1/B2 remain untouched.

Original cmerc-repro / b4a3138 / accepted ECERC data checks passed. Same
Flat1600 + Local Skip, lr1e-5,Adam,50epochs,batch32,GCNet random-mask-v1.
61 new/legacy orchestration checks passed; no new GPU smoke or altered model.
Audited original worker checks initialization and actual train/evaluation mask
streams against O6 seed66, records actual optimizerLR/WD everyepoch, preserves
8BEST+exact selected predictions/masks+full last_training recovery state.

Completed67/68 results are reused read-only. Finite controller automatically
writes all15 runs including3 existingWD2e-4 controls, five three-seed means/
sample standard deviations and failures after these4 finish. No new search.
LAUNCH_STATUS.json is a timestamped snapshot, not a claim that50epochs finished.

Actual command from immutable snapshot, via tmux:

```
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_meld_wd_seed66_20261018/run.py \
 --output /home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1 \
 --driver-commit 3628d1e69d611f7d6a8a0d6896b435a63f805a9c
```
