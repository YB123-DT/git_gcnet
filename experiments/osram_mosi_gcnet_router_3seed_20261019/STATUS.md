# Status: six MOSI training jobs running

Latest verified launch: see LAUNCH.json for exact UTC, PIDs and source hashes.
All six workers have completed3–7/150epochs with `gcnet-random-mask-v1` in
their effective configs. GPU7 carriesseed66, GPU1seed67, GPU3seed68; each has
one Flat and one Joint worker. No GPU4 use.

- Immutable numerical source:38646302489b1ce74cf2d4ddeb9a9ec0ae5eb49d.
- tmux:mosi_gcnet_3864630; queuePID2978050.
- PIDs:seed66Flat2979821/Joint2979906;seed67Flat2980012/Joint2980105;
  seed68Flat2980195/Joint2980318.
- Output:/data2/yb/remote_experiments/osram_mosi_gcnet_router_3seed_20261019/execution_v1.
- Actual s0 GPU smoke passed: finite task/backbone/Router gradients and parameter
  updates, full model/Adam/RNG reload. Allocated peak3092.058MiB,3training batches.
- Current-source legacy checkpoint bridge reproduces predictions, labels and
  availability EXACTLY. New runs then discard these weights, initialize fresh,
  and switch to GCNet masks. This bridge is not a new model performance result.
- Allthree seed pairs have exact initial backbone hashes and evaluation-mask
  hashes. Full training-mask and final metric checks run at queue completion.
-77local regression tests passed, compilation anddiff checks passed. The only
  local warning concerns pre-existing optional torch-sparse/libcusparse import.

The upstream missing DATA.json was recovered byte-identically from the tracked
complete150 manifest; all6598actual file hashes passed. Old files were not
recreated or overwritten. Previous not-started notes below are historical.

User approved six fresh MOSI150 runs with GCNet missing on2026-10-19.
The earlier legacy-mask MOSI preflight failure remains a separate record.
No results from that failed queue or historical80.747 baseline are relabelled
as GCNet-missing results.

Worker correctness checks, real-data smoke and immutable deployment have passed.
Formal training is in progress; no final scores yet.
