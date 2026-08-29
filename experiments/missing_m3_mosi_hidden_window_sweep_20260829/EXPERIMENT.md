# CMU-MOSI Hidden–Window Sweep

## Locked protocol

- Model: Slot Missing-M3, regression, both graph branches;
- Training: all-rates-per-batch, 100 epochs, batch size 32;
- Fixed: `time-attention=False`, LR `5e-4`, L2 `1e-5`, JEPA weight `0.1`;
- Grid: hidden `{50,100,200}` × symmetric window `{1,2,3,4}`;
- Stage 1 seeds: 66, 67, 68;
- Selection: three-seed validation eight-rate W-F1 mean only.

## Launch record

The 36-job matrix passed local and remote tests. GPUs 0, 2, and 3 were occupied by
unrelated 24.8 GB processes, so the non-conflicting free GPUs 5, 6, and 7 were used:

| GPU | Seed | Concurrent jobs |
|---:|---:|---:|
| 5 | 66 | 12 |
| 6 | 67 | 12 |
| 7 | 68 | 12 |

Each task uses two Torch CPU threads. At steady startup the cards used approximately
20.4, 24.9, and 22.3 GB respectively, confirming that twelve jobs fit on each V100.

One of 36 simultaneous cuDNN initializations returned
`CUDNN_STATUS_NOT_INITIALIZED`; the other 35 jobs continued. A detached retry watcher
will rerun every output directory lacking `metrics.json` after the first wave exits,
without duplicating completed jobs.

Remote result root:

```text
/data2/yb/remote_experiments/missing_m3_mosi_hidden_window_sweep_20260829
```

Results remain pending and no performance claim is made here.
