# Verified submission state, 2026-10-17

Formal training has NOT started. Do not report the retry helper as a GPU job.

- Source/adapter/controller commit:`c8630e7d269fda63ba4d1ef008787d68f096698b`.
- Immutable local submission snapshot:
  `/data2/yb/paper/feature_sources/alternate_flat_code_c8630e7`.
- Local tmux session:`meld_alt_features_submit_1017`.
- Local state/log directory:
  `/data2/yb/paper/feature_sources/alternate_flat_submission_20261017`.
- Latest verified first attempt:`2026-10-17T06:07:19Z`,
  `status=awaiting_ssh_or_transfer`, `formal_training_started=false`.
- Actual SSH error:`Connection timed out during banner exchange`,port22.
  Several separate read-only checks also returned connection resets/timeouts.
- Intended server:`ssh biggpu`,Python
  `/data2/yb/conda_envs/cmerc-repro/bin/python`,unchanged.
- Worker file content SHA:
  `ad422dce4c1c1d2c79cea5e07c7cbdaf405bb85009e64f8f4274c492a339a3c8`.

The helper is running persistently and retries at60second intervals,up to1440
attempts. It may submit exactly one complete MAGTKD feature-source Flat seed66
job after environment/resource checks and data/code transfer. It cannot submit
MM-Align while coverage is incomplete,and cannot switch to local training.
The remote chain executes a separate three-batch smoke before the50epoch run.
No formal score or successful GPU smoke is available at this checkpoint.

Verification before submission:14CPU tests passed,compilation/whitespace passed;
all canonical speakers,labels,sentences and split sequences match after feature
replacement. The existing loader reads1432dialogues/13708utterances and
A/T/V dimensions768/768/768. No package installations or model-code changes.

Actual command (local CPU submission helper only):

```bash
tmux new-session -d -s meld_alt_features_submit_1017 \
  '/data2/yb/conda_envs/cmerc-xp-repro/bin/python -u /data2/yb/paper/feature_sources/alternate_flat_code_c8630e7/experiments/osram_meld_alternate_features_20261017/submit.py --data /data2/yb/paper/feature_sources/converted_20261017/magtkd_meld.pkl --state /data2/yb/paper/feature_sources/alternate_flat_submission_20261017 --controller-sha c8630e7d269fda63ba4d1ef008787d68f096698b > /data2/yb/paper/feature_sources/alternate_flat_submission_20261017/submission.log 2>&1'
```

MM-Align source remains blocked:official Drive quota interrupted the archive at
838860800/2293780480bytes; only dev/test/embedding are fully retrieved. Dev is
missing canonical110_7. No samples were dropped,filled,relabeled or mixed with
old ECERC features. User clarification was requested for any reduced-subset
comparison; no such expansion is currently submitted.

## User-requested immediate download retry, 2026-10-17

Ran the existing resumable official downloader again in the dedicated tmux
session `mmalign_download_resume_1017`,using the same partial archive and URL.
The fresh request failed with explicit `QuotaExceeded: Google Drive quota
exceeded; partial archive preserved for a later resume`. The session exited;
there is no active MM-Align download to report. Archive size remains exactly
838860800bytes. Retry log:
`/data2/yb/paper/feature_sources/mmalign_20261017/resume_20261017.log`.

A fresh `ssh biggpu hostname` again failed at SSH banner exchange. The separate
MAGTKD bounded automatic SSH/submission helper remains alive; no formal GPU
training has started. No code,environment,seed,missing protocol or samples were
changed. Existing data are preserved. Official provider quota/server recovery
or an accessible complete official feature copy is required for further progress.
Do not call the quota failure a slow-but-running download,or start MM-Align
training on a partial archive.
