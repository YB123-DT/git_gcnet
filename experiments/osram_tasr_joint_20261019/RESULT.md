# Completed one-stage TASR, MELD seed66

INTERNAL TEST-ORACLE SCREEN. Read directly from the remote completed SUMMARY
and original seed66 Flat SUMMARY at2026-10-19T02:50:29UTC. No new inference,
training, mask resampling or parameter changes were made for this report.

| Method | Random8 ACC | UA | W-F1 | High ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original Flat seed66 |61.786398|38.420205|58.889822|57.931034|32.489727|53.699450|
| Joint TASR seed66 |61.786398|38.420205|58.889822|57.931034|32.489727|53.699450|

All eight reported classification metrics and selected epochs match the
original Flat seed66; not merely their rounded means. CE has small numerical
differences (Random8 TASR1.1955178662 vs Flat1.1955178196). Therefore the
currently verified conclusion is NO classification metric gain in this run.
Equal aggregate metrics do not by themselves prove identical sample predictions,
weights, or an inactive module. Do not infer a mechanism without those checks.

| Rate | ACC | UA | W-F1 | BEST epoch |
|---|---:|---:|---:|---:|
|0.0|67.509579|44.329549|65.691323|8|
|0.1|65.708812|45.581320|64.245048|16|
|0.2|64.099617|40.927724|61.879530|11|
|0.3|62.260536|40.207227|59.995707|21|
|0.4|60.919540|38.846640|58.208615|35|
|0.5|59.770115|33.187929|55.672351|13|
|0.6|57.432950|32.255425|53.141099|24|
|0.7|56.590038|32.025827|52.284899|46|

The accepted61.557/37.728 historical baseline is a THREE-seed mean, not the
seed66 baseline above. Comparing joint TASR seed66 to that mean would incorrectly
suggest a gain that is absent in the matched seed66 classification results.

## Completion and protocol

- Statuscomplete,50/50epochs, no reported error; process117308 exited.
- Eight BEST checkpoints and complete last.pt verified present.
- Seconds1616.2317 (about26.94minutes), peak allocated1404.2988MiB.
- One fresh joint model, all13,350,218 parameters trainable, including59,423 TASR.
- Adam LR1e-5/wd1e-5/batch32; original random-only CE and GCNet missing.
- Last three epochs each had33 forwards and33 optimizer steps; no second view.
- Requested per-rate test ACC-best selection, same-forward UA/W-F1/masks.
- Training sourcec5c26dd; biggpu physicalGPU1, cmerc-repro. Existing artifacts
  and other users' processes were not modified.
- This is eight separately selected checkpoints from one training trajectory,
  NOT one deployable checkpoint's independent generalization score.

## Evidence locations and current verification limit

Joint source:
`/home/qinjinghui/remote_experiments/osram_tasr_joint_20261019/runs_v1/MELD_seed66/SUMMARY.json`.
Flat reference:
`/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1/S66_WD1/SUMMARY.json`.
Joint history, best_miss_*.pt, last.pt, selection_versions/*.npz and separate
paired_mask_replay remain remote. All selected artifact SHA256 and exact mask
hashes were present in the successfully read SUMMARY. This report uses those
recorded results, not a freshly resampled evaluation.

Subsequent attempts to download the full SUMMARY and verify saved prediction
arrays/learned weights failed during SSH banner exchange. They did not execute
remote changes, rerun inference, restart training, or kill a process. Thus the
additional sample-level equality and final Adapter-weight checks are NOT yet
verified here. Initial real GPU gradient/update/recovery evidence remains in
SMOKE.json, but it cannot replace checking the trained final branch contribution.
Persistent and paired replay values are not invented or inferred from random
metrics. No additional seed, LR search or experiment was launched.
