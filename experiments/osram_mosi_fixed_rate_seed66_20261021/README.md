# MOSI independent fixed-rate Flat, seed66

INTERNAL TEST-ORACLE SCREEN

Eight independent original Flat models, one designated missing rate each:
0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7. Train rate equals test rate; not cyclic.
No new Block, source tracker, Query, Memory or additional loss is enabled.

Each run:150 epochs, original s0 environment/data split, seed66, batch32,
Adam, weight_decay1e-5, MSE, LR3e-4 for100 epochs then3e-5 for50.
Original gcnet-random-mask-v1 semantics remain unchanged. Only training-rate
selection changes. Selected Test-best W-F1 predictions/masks are saved from
the same call as their metrics, along with best.pt and full-state last.pt.
Old cyclic baseline, V3 and Nested jobs/results are not modified.

## Persistent queue

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosi_fixed_rate_seed66_20261021.run \
  --output /data2/yb/remote_experiments/osram_mosi_fixed_rate_seed66_20261021/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4
```

Root validates real-GPU endpoint behavior before deployment. The queue does not
add another smoke job: rate0 starts first; after its first measured epoch,
up to four jobs run concurrently using >=4GiB/job (or1.4× measured peak if larger),
2GiB GPU reserve, >=64GiB available host RAM and >=15GiB disk headroom.
Unknown/bad GPU4 is rejected. No processes are killed. A failed worker stops
new submissions while already running peers finish. Fresh output directories,
exclusive lock, PID/command/log records and persistent QUEUE.json avoid repeats.

Each output lives under runs/rate_0pX/seed_66. The queue publishes RESULT.md,
REPORT.json and METRICS.csv from completed summaries only. Eight-rate and
high-missing means require completion of all rates in that scope. The resulting
mean describes eight independently trained models, not a single deployment model.
Each model gets300 optimizer updates, matching the cyclic model's150epochs.
Across all8 models the budget is2400 updates, eight times one cyclic trajectory.
This tests rate specialization, not an equal-total-training-budget improvement.
No automatic seeds, HPO, retry, or changes to batch size/precision are allowed.
