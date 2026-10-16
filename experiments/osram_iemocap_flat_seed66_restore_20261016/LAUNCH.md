# Flat-only launch evidence

INTERNAL PER-RATE TEST-ORACLE ACC — TRAINING IN PROGRESS, NOT FINAL SCORES.

The user's final scope is **original Flat only**, seed66, IEMOCAP-4 and
IEMOCAP-6, one complete fivefold run each. Ten fold jobs, 100 epochs each;
no B6 rerun or extra seed. Original Local Skip remains enabled. The original
corrected `gcnet-random-mask-v1` and cyclic0.0–0.7 protocol are unchanged.

- Implementation commit: `853e0c8ee0830db3f7fc855e63268020b9cbcd0e`.
- Immutable numerical source: `785a01242ef1a82b97d5337acbb33ba963df5537`.
- Server: `ssh biggpu`.
- Implementation checkout: `/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_code_853e0c8`.
- Output: `/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016`.
- tmux: `iemocap_flat66_restore_1016`; controller PID at launch: `2852840`.
- Local targeted verification: **20 passed**; one existing optional torch-sparse import warning.

Verified after launch: five workers running, five queued, no failed jobs.
GPU1 runs four workers; GPU3 runs one within its available memory. No task
was added on GPU4 or GPU5 and no unrelated process was signalled.

| Physical GPU | UUID | Initial worker PIDs |
|---|---|---|
| 1 | `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049` | 2852880, 2853020, 2853120, 2853152 |
| 3 | `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a` | 2852914 |

Actual first-epoch retention checks already passed for IEMOCAP-4 folds5/1/2
and IEMOCAP-6 fold1. Each check reads eight saved BEST checkpoints and their
matching selected-epoch artifacts, verifies epochs, hashes, predictions and
ACC/UA, and confirms one saved CUDA RNG state. These checks add no random
evaluation or model forward. Other folds must pass the same gate before
continuing beyond their first epoch. No final ACC/UA is available yet.

Reproduction command, launched from the immutable implementation checkout:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_iemocap_flat_seed66_restore_20261016.run \
  --worker /home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_code_853e0c8/experiments/osram_iemocap_b6_transfer_20261015/worker.py \
  --smoke-summary /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/smoke_v2/SUMMARY.json \
  --data-provenance /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/DATA_PROVENANCE.json
```

The previous real longest-batch smoke bounds memory admission; no additional
standalone GPU smoke was run. `MANIFEST.json` contains effective configs,
source/worker/data hashes and the reused smoke provenance. `queue/QUEUE.json`
and `RUNNER.json` are the live status authorities. `controller.log` and
`queue/logs/` contain logs. Completion requires successful worker summaries
and retained selected artifacts, not merely process exit.

The queue retains all80 rate-selected BEST weights and matching masks/logits,
plus ten complete `last_training` states. Fivefold results are only reported
after all five folds of a dataset finish successfully. This new run cannot
recover previously discarded historical weights or promise identical old
scalar scores.
