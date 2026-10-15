# Completed GPU3/5 training: exact Test-best recovery

INTERNAL TEST-ORACLE THREE-SEED REPLICATION.

Flat_matched seed67(GPU3)and seed68(GPU5)both completed all50epochs and
retained all8BEST checkpoints plus their exact selected-epoch artifacts.
They then failed in the optional common-mask replay, not in optimization.

Root cause: the inherited seed66 screen `paired_replay` scatters a saved mask
positionally and compares labels by cursor. Official test loaders use a
master-seed-derived conversation sampler, so67/68enumerate the identical test
set in another order. The assertion correctly rejects the invalid positional
comparison. This defect was previously documented in the older three-seed
replication's POSTPROCESS_FIX.md; this new launcher's use of full replay failed
to carry forward that known restriction. This is an integration oversight,
not evidence that the missing sampler was changed or training weights lost.

Do not restart training,change weights,reselect checkpoints,resample test masks,
or remove the label assertion. Recover requested Test-best scores on CPU using
the unchanged frozen source's existing `recover_testbest.recover_run`:

* Verify all50completed epochs and all8checkpoint-selected artifact hashes.
* Verify complete `(conversation_id,utterance_index)`key sets,label/speaker per
  key,legal GCNet masks and both rate0/.7endpoint conditions.
* Recompute ACC/UA/WF1from the original selected predictions,confirm ACCagainst
  each checkpoint's stored selection_accuracy. No new inference or threshold.
* Use an ID-bearing `paired_miss_0p0.npz`from the chosen seed66trial11as the
  canonical identity reference. The older COMMON mask-only archive has noIDs
  and must not be substituted for this check.
* Preserve original failed worker summaries,queue status and logs. Write new
  reports separately under `testbest_recovery_v1/`. Mark common-mask replay as
  not completed; never put these Test-best values in a paired-mask column.

Verified recovered8-rate scores:

| Configuration | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| Flat_matched | 67 | 60.320881 | 36.160538 | 56.566342 |
| Flat_matched | 68 | 60.804598 | 35.644525 | 56.734708 |

These are two completed seeds,not a three-seed mean. Flat66and the four B6
replications were still training when this repair started.

`recover.py`continues the same CPU-only repair for the seven approved jobs as
they finish. It rejects incomplete training,missing checkpoints and unrelated
errors rather than turn arbitrary failures into successes. It preserves already
successful complete summaries,including any genuine paired replay. It does
not submit training jobs,allocate CUDA memory or change the immutable training
source. A separate lock prevents duplicate repair managers and a12hour bound
prevents indefinite polling. Five bounded recovery/configuration tests passed.

```bash
CUDA_VISIBLE_DEVICES='' /data2/yb/conda_envs/cmerc-repro/bin/python -u \
  /home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015_recover.py \
  --source-root /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57 \
  --launcher /home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015_run.py \
  --root /home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015 \
  --wait
```

Read final requested metrics from `testbest_recovery_v1/RESULT.md` and
SUMMARY.json/CSV,not solely the original auxiliary-replay failure state.

Observed live CPU-only recovery manager PID2382405; both Flat67/68recovered
successfully and remaining five jobs still training. The committed
RECOVERED_PROGRESS.md/CSV and RECOVERY_PROGRESS.json preserve this partial
snapshot,not final three-seed completion. At that check Flat66was23/50epochs
and the four B6runs10/50. No additional GPU inference was launched.
