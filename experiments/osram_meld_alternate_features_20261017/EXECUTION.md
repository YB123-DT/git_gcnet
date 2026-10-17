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

## SSH boundary diagnosis

Fresh `ssh -v -o BatchMode=yes -o ConnectTimeout=8 biggpu hostname` established
TCP and sent the local OpenSSH version string,but timed out before receiving
the server banner. A separate direct TCP port22check connected in0.005seconds
and received no SSH banner within5seconds. Authentication was never reached.
This does not identify a key/password failure,prove the server is down,or
establish whether earlier training processes are still alive. SSH service,
host-resource and network-policy causes are not separated without another
client or console access. No server service/process was modified or stopped.
Next safe check is an independent user connection; if that also fails,an
administrator needs an existing session/console for host and SSH-log inspection.

## SSH recovery and bounded restart, 2026-10-17

SSH is reachable again. The automatic submitter had submitted the MAGTKD job
at06:24:42UTC on healthy physicalGPU3. Its real three-batch smoke completed
(11.977seconds,611015680bytes peak allocation). Formal attempt1 failed after
epoch1 at the strict RNG recovery guard. No complete `last_training.pt` exists;
all initial artifacts are preserved under `runs/Flat_MAGTKD_seed66` and are
excluded from valid model ranking. This is not a data/model failure diagnosis.

Root cause was reproduced directly:CUDA runtime1device,patched public Python
count1,but `torch.cuda.random.device_count()` still0because Torch1.13captured
the old function at import. `get_rng_state_all()` therefore returned0states
despite an initialized CUDA context. Synchronizing the random-module alias
changed RNG count to1without removing the checkpoint guard.

Worker fix:`6392ff5c31ebaa75d485bb0ff426146d2fce072b`; fileSHA
`66fc911607d06ebc98b7e7ab407aaf786185586eebcd8cb2ac4ab4e6b46a48d1`.
Only the admitted single-device process-local compatibility shim changed.
Core131source-file inventory,model,task loss,data,mask and optimizer settings
remain fixed. The fix also makes CUDA seeding enumerate the real admitted
device; do not claim bit-identical RNG behavior with the faulty attempt.

Regression:the captured-alias test failed before the fix; all15CPU tests pass
after it. A tiny realGPU optimizer/save/restore probe passed,including exact
CPU/CUDA random replay and exactly1saved CUDA RNG state. Evidence:
`rng_recovery_probe_v2/RESULT.json` remotely and `RNG_RECOVERY_PROBE.json` locally.
No new full three-batch model smoke was needed; the original completed smoke
plus the targeted real recovery probe are explicitly distinct.

The corrected formal run starts from scratch at
`runs/Flat_MAGTKD_seed66_v2`,PID1365346,tmux
`meld_flat_magtkd_seed66_v2_1017`,same physicalGPU3 UUID and cmerc-repro Python.
On first verification the worker reportedrunning,0completed epochs,no error,
runtime/public/random counts1/1/1. Log:`pipeline_v2.log`. New source/config/data
are not mixed into the preserved failed attempt. Await first committed epoch
before claiming actual full-state training recovery works on this model.

Subsequent fresh verification confirms1/50epochs completed in v2,no error,
`last_training.pt` committed atnext_epoch1with1CUDA RNG state and72optimizer
parameter-state entries. This establishes the previously failing save boundary
now passes for the actual MAGTKD Flat training,not only the tiny recovery probe.
The run remains active. Final ACC/UA results are pending.

Any future strictly paired ECERC control must use the same corrected worker
compatibility shim; this experiment does not retroactively make historical
wrappers' CUDA seeding or checkpoints bit-identical,and no extra control run
is authorized or launched by this repair.

## Interim progress, 2026-10-17 06:58:37 UTC

Remote v2 SUMMARY reports running,no error;history contains21completed epochs.
PID1365346 remains the intended MAGTKD Flat worker. Recomputed ACC/UA from the
eight exact saved selected-epoch artifacts,without new inference. All ACCs
matched checkpoint metadata within1e-8fraction units. Eight-rate mean is
ACC54.583333%,UA31.867129%; per-rate scores and selected epochs are in RESULT.md.
Epoch21training W-F1 is0.9210;its ordinary test mean ACC is0.5184 (distinct from
the accumulated per-rate Test-best mean). This train/test gap is an observation,
not a proven feature mismatch or causal diagnosis. No configuration was changed.
MM-Align remains blocked by the recorded official quota/full-coverage issues.
