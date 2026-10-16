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
