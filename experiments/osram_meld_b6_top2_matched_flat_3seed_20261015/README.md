# MELD: two selected B6 configurations and a matched Flat, three seeds

INTERNAL TEST-ORACLE THREE-SEED REPLICATION. User-approved finite continuation,
not a further HPO study or an independent paper test estimate.

## Fixed comparison

| Configuration | Existing seed66 ACC / UA | New training seeds |
|---|---|---|
| B6_default: rank2/steps14, threshold_init0.1, B6LR multiplier1 | 61.2165 / 35.9354 | 67,68 |
| B6_tuned: rank2/steps14, threshold_init0.33599569865951234, multiplier2.8200380294401013 | 61.1925 / 36.0079 | 67,68 |
| Flat_matched: same main training settings, no Block2 | not yet measured | 66,67,68 |

Exactly seven new independent trainings; the two selected B6seed66 results are
reused, not resumed or counted as newly trained. Existing accepted Flat baseline
ACC60.693/UA36.255/WF57.045 uses other training hyperparameters and remains
preserved. It is not overwritten, nor used as this matched control's score.

All seven jobs: main AdamLR0.0003433234536467373,WD0.0000044857894231052665,
dropout0.22489055023388654,batch32,50epochs,official MELD splits/features,
SHA256a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80.
Retain gcnet-random-mask-v1,cyclic0.0–0.7,emotion-only,no-JEPA,original CHM
joint write and queries,Flat1600and rawLocalSkip. No new loss,module,missing
schedule,early stopping or model search. FlatB6LR multiplier stays1,not the
tuned candidate's multiplier. Both B6hyperparameters are fixed across seeds.

Per-rate Test-best ACC retains its exact checkpoint,predictions,labels and masks;
UA/WF1 come from those same selected predictions. Also retain original protected
common-mask replay separately. Do not put paired-mask UA beside Test-best ACC.
Eight selected checkpoints are not one deployment checkpoint. Report all seeds,
8-rate and high-missing metrics,per-rate scores,mean ± sampleSD,failures and
source provenance. Do not calculate a three-seed ranking from incomplete seeds.

## Source and compatibility

Every new worker uses immutable training commit
`9fa9f571e7095d5b09f9c256d888c31a76c82744` at
`/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57`.
The standalone launcher is versioned separately and never edits that checkout.
Seed67/68 use the original worker's explicit replication seed guard; Flat66
uses the normal seed66 entry. Full Test-best and paired replay remain enabled.

Default B6seed66 originally used9048ffd; tuned seed66used9fa9f57. Preserve these
different original provenance records rather than claim all nine were rerun at
the new source. Independent read-only compatibility audit dynamically loaded
the actual9048ffd implementation and verified rank2/14default parameters,
decomposition forward,input gradient and threshold/step gradients exactly equal
to the new .1default implementation. Default multiplier1 preserves optimizer
membership/order; RNG-isolated initialization and SVD init clipping are unchanged.
Fourteen related CPU tests passed. This supports default computation
compatibility,not an assertion of bitwise identical full training or a silent
substitution of a lower-scoring rerun for the user-selected seed66.

Existing recovery caveat: prior checkpoints contained empty CUDA RNG lists.
This task does not modify numerical training source to fix that issue. The
launcher adopts still-live owned workers on restart, but refuses to auto-resume
dead workers with incomplete saved CUDA RNG or silently treat weights-only
loading as complete recovery. It fails visibly and records the limitation.
Never kill other users' tasks,delete checkpoints,or change batch size to fit.

## Resource plan and execution

Server:biggpu. Python:`/data2/yb/conda_envs/cmerc-repro/bin/python`.
Use GPU2up to5jobs,GPU3one job,GPU5one job. All device admissions use fresh
physical index→UUID mapping,exclude GPU4,observe other queues/processes and
retain the six-total-compute-processes/card safety ceiling. New jobs use unique
directories. Measured prior same-architecture real-batch allocated peak was
1004MiB; use2048MiB/job conservative admission reservation plus existing margins,
not a falsely claimed new memory measurement. No new GPU smoke is needed for
unchanged numerical models,optimizer and already-tested seed guard.

Output:
`/home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015`.
Standalone deployed launcher:
`/home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015_run.py`.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  /home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015_run.py \
  --source-root /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57 \
  --output /home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015
```

Run `--check-only` first for source/dataset/config/GPU/disk preflight. A whole
controller lock prevents duplicate submissions. Existing queue PID/command/start
identity checks adopt live workers rather than launch duplicates. The existing
protected Flat24checkpoints and previous studies are never modified.

Final `RESULT.md`, `SUMMARY.json` and `SUMMARY.csv` are generated automatically
after all seven jobs reach terminal states. Until then RUNNER/queue/live logs
are the authoritative running status,not a completion claim.

## Post-training replay defect and recovered result location

GPU3/5Flat67/68finished50epochs but inherited cross-seed positional replay
failed. Training and exact selected artifacts are intact. Requested Test-best
metrics are recovered separately on CPU,with ID-keyed label/speaker checks and
artifact/checkpoint hashes; no retraining,new inference or new mask selection.
See POSTPROCESS_FIX.md. The repaired seven-job continuation writes its final
RESULT.md and SUMMARY.json/CSV under `testbest_recovery_v1/`. Original failed
worker/queue summaries remain preserved; common-mask replay is explicitly not
completed for repaired67/68runs. Do not report the two evaluation scopes as one.

## Final requested Test-best comparison

Training and Test-best replication complete; see RESULT.md,SUMMARY.json/CSV and
FINAL_CHECKS.json. Seven new50epoch runs finished,all nine seed results have
eight rates,and all72new/reusedBEST checkpoint files are preserved remotely.
Three-seed means(ACC/UA/WF1):Flat_matched60.635/36.304/56.859;
B6_default60.827/35.518/56.801;B6_tuned60.809/36.192/56.939.
Neither B6candidate improves averageACCandUAjointly. Keep original auxiliary
failure states and the explicit incomplete common-mask-replay caveat. No new
methods,seeds,training runs or HPOare automatically authorized by completion.
