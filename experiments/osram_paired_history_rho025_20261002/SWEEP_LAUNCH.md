# Ten-run rho sweep — launched

User authorized ten concurrent independent seed66/100epoch task-only jobs.
GPU5:C01–05 rho=.05,.10,.15,.20,.25; GPU6:C06–10 rho=.30,.35,.40,.45,.60.
No InfoNCE contribution, drop=.2, all other A settings unchanged.
Thirteen paired-view tests passed; ten configs validated and differ only in rho.

Code commit6e2a998. Server biggpu, isolated snapshot and runs:
`/data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/{code,runs}`.
Use /data2 because /data1 had only7.7GB free versus /data2 626GB.
GPU4 excluded. Both assigned cards initially empty; UUIDs pinned in sweep.py.
Launch metadata/PIDs/commands retained in sweep_launch.json. Children each have
independent sessions, logs, config, source hashes and PROVENANCE.json.

All ten processes verified alive concurrently after launch. Runs were staggered
four seconds to inspect remaining memory, not queued until earlier runs finish.
Training checkpoints remain per-rate BEST; reference Flat and old rho.5 A reused.
Existing loss summary for A/B retains their .5 formula, not used for this sweep.
New logs record task_view2_weight; interpret task as (1-rho)*View1+rho*View2.

Status: RUNNING, not completed. Results are internal Test-oracle diagnostics;
do not report the best test-selected rho as validation-selected paper performance.
