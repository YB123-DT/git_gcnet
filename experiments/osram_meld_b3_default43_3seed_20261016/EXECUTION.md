# Verified launch, 2026-10-16

INTERNAL TEST-ORACLE THREE-SEED REPLICATION. Running,not completed.

- Controller source commit:`033cee4372af0998f48e4e97560062c188a952de`.
- Numerical worker source:`737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`.
- Server:`ssh biggpu`; Python:`/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Physical GPU3,UUID`GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`; worker-visible cuda:0.
- tmux:`meld_b3_default43_3seed_1016`; controller PID3058359.
- Controller deployment:
  `/home/qinjinghui/remote_experiments/osram_meld_b3_default43_controller_20261016`.
- Output:
  `/home/qinjinghui/remote_experiments/osram_meld_b3_default43_3seed_20261016`.

At2026-10-16T13:31:17UTC,the persistent queue and both worker summaries reported
running. Both GPU processes were present on GPU3; no completed epoch yet.

| Seed | PID | Observed GPU allocation MiB | Status |
|---:|---:|---:|---|
|66|reused|not rerun|complete:ACC60.756705/UA37.357830|
|67|3059135|1660|running,first epoch|
|68|3059305|1640|running,first epoch|

Another pre-existing GPU3 process was left untouched. GPU4 is excluded. No
duplicate default4/3seed67/68 queue/output was found before scheduling. Exactly
two new runs; all configs deep-copy the complete original66config and change
only seed. Explicit worker arguments include `--testbest-only` and
`--replication-seed 67`/`68`. The worker cwd is the unchanged original737checkout.

Preflight passed:all129recorded source-file hashes match,all eight reused66BEST
weights exist,MELD feature SHA matches,disk had134GiB available,physical GPU UUID
matched,and the original larger8token/4round smoke was reused (1886.161MiB).
No new GPU smoke,model edits,common-mask replay or hyperparameter search.
The existing optional torch-sparse import warning is present; no traceback or
failed worker was observed. Neither this warning nor launch proves performance.

Regression checks:two tests passed for exact seed-only copies and rejection of
changed structure/LR/missing/LocalSkip/source. Compilation and whitespace checks
passed. A bounded independent read-only audit verified immutable source and
worker seed flags. No training was delegated to or duplicated by the auditor.

Actual persistent command:

```bash
tmux new-session -d -s meld_b3_default43_3seed_1016 \
  '/data2/yb/conda_envs/cmerc-repro/bin/python -u /home/qinjinghui/remote_experiments/osram_meld_b3_default43_controller_20261016/experiments/osram_meld_b3_default43_3seed_20261016/run.py --controller-sha 033cee4372af0998f48e4e97560062c188a952de > /home/qinjinghui/remote_experiments/osram_meld_b3_default43_3seed_20261016/controller.log 2>&1'
```

Final `RESULT.md`, `SUMMARY.json` and `SUMMARY.csv` are generated automatically
after both new workers finish. Preserve BEST checkpoints and exact selected
prediction artifacts; no final three-seed improvement is claimed at launch.

## Completion verified: 2026-10-17 03:00:38 UTC

Controller and both new workers are complete. Seeds67/68 each have50history
records and no reported error. All24BEST weights (eight reused66plus16new)are
present remotely. All24exact selected-prediction SHA256 hashes match their
summaries. ACC and seven-class UA were independently recomputed from every
saved artifact and match within1e-8. Three-seed means/sample SD also match
independent recomputation within1e-10. No training or inference was added.

Final three-seed eight-rate means:ACC60.553959 ±0.201168,
UA36.486390 ±1.411878,W-F1 57.011331 ±0.520243 (sample SD).
High-missing means:ACC56.585781,UA29.835688,W-F1 51.065596.
`RESULT.md`, `SUMMARY.csv` and `SUMMARY.json` are archived alongside this record.

Relative to the user-accepted historical Flat reference60.693/36.255/57.045,
the approximate changes are -0.139ACC,+0.231UA,-0.034W-F1 percentage points.
The source snapshots and randomly sampled winning-epoch masks are not shared
between that baseline and B3; this is a descriptive reference comparison,not
an exact common-mask paired experiment. The accepted baseline is unchanged.
UA gains are not seed-consistent:seed66is higher,67and68are lower than their
accepted Flat rows. This does not establish stable joint ACC/UA improvement.
The authorized two-run extension ended; no new HPO or seeds were submitted.
