# CMERC sentiment three-seed queue

INTERNAL TEST-ORACLE SCREEN. This finite queue trains only MOSI seeds67/68 and
MOSEI seeds66/67/68; MOSI66 reuses the complete control in
`/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`.

Use an immutable checkout of numerical source
`0da38b3326fdad09da0195e49b8b7d6424b4f613` with this experiment directory overlaid.
Start from that checkout using `/data2/yb/conda_envs/cmerc-repro/bin/python -m
experiments.osram_cmerc_sentiment_3seed_20261023.run --source-commit RUNNER_COMMIT --output /absolute/fresh/output`
inside a persistent tmux session. The controller uses its checkout ROOT as worker
cwd and records QUEUE.json, separate run logs and outputs. GPU7 UUID is preferred;
physical GPU4 and its UUID are forbidden. Admission requires at least6GiB GPU
free (4GiB budget plus2GiB margin),64GiB RAM and20GiB disk, at most5 children,
with4seconds between starts and a fresh resource check before each start.
`--source-commit` identifies the committed runner overlay; QUEUE.json separately
records the pinned numerical source commit above.

All jobs use original Flat/newGCNet missing, batch32, Adam WD1e-5, MSE and
per-rate Test-best W-F1. MOSI uses150epochs, LR3e-4 for100epochs then3e-5;
MOSEI uses60epochs at constant1e-4. This is an internal oracle screen, not an
independent-test claim or a single-checkpoint deployment result.

Resume only with explicit `--resume` and the same queue identity. A surviving
recorded child prevents duplicate submission; interrupted training requires
`last.pt` and the worker validates complete training state. Failed jobs are not
retried. Child failure drains already-running peers and stops pending jobs;
the controller never kills another process. Do not touch unrelated queues.

Only after all five jobs and the reused MOSI66 pass selected-checkpoint and
prediction integrity validation does reporting write RESULT.md, RESULT.json,
and PER_RATE.csv, with all three seeds, per-rate metrics and mean/sample SD.
No partial means or additional inference are used.
