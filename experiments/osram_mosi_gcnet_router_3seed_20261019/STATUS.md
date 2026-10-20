# Status: six MOSI runs complete; results verified

2026-10-20: allsix runs finished150epochs. Final single-validation-checkpoint
metrics and selected epochs are inRESULT.md; fullper-rate table inMETRICS.csv.
SUMMARY.json preserves compact run/source/resource details and pairing checks.
All48prediction artifacts passed recordedSHA256 and independently recomputed
ACC/W-F1 checks (maximum error0pp). No weights or large prediction files are
copied intoGit. Completed queue/source remain onbiggpu at the recorded paths.

Three-seed eight-rate mean:Flat ACC76.372/W-F176.422 versusJoint
ACC75.889/W-F175.788. High-missingW-F169.554→67.735. This does not support
a stable MOSI benefit. Original80.747 had BOTH different masks and per-rate
Test-best selection, so the total difference is not a missing-sampler-only effect.
No new training, inference, tuning, or cleanup was performed for this report.
Earlier startup progress below is historical, not current state.

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
