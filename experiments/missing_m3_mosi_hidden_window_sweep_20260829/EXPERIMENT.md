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

Each task uses two Torch CPU threads. Twelve jobs initially entered each V100, but
later dynamic batches exposed seed-dependent peak memory: GPU5 reached about 30.8 GB,
and multiple jobs exited with CUDA OOM or `CUDNN_STATUS_NOT_INITIALIZED`. Therefore
startup memory was not a valid capacity test.

The surviving jobs continue without interruption. The original all-at-once retry
watcher was removed. A tested replacement waits for the initial runner to exit, skips
directories containing `metrics.json`, and reruns the remaining jobs in four waves
with at most three concurrent jobs per GPU. This avoids duplicating successful jobs
and prevents the observed peak-memory failure from repeating.

Remote result root:

```text
/data2/yb/remote_experiments/missing_m3_mosi_hidden_window_sweep_20260829
```

## Final three-seed results

All 36 jobs completed. Values are percentages averaged over seeds 66--68.

| Hidden | Window | Validation 8-rate | Test miss0 | Test 8-rate | High 0.4--0.7 |
|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 78.4104 | 85.3484 | 78.0280 | 73.8488 |
| 50 | 2 | 78.1977 | 85.1913 | 77.7931 | 73.6669 |
| 50 | 3 | 77.4383 | 85.3849 | 77.7293 | 73.4806 |
| 50 | 4 | 76.9687 | 85.5873 | 78.6482 | 74.6391 |
| **100** | **1** | **78.4999** | 85.2483 | **78.8965** | 74.8143 |
| 100 | 2 | 77.8421 | **85.6919** | 78.4405 | 73.9987 |
| 100 | 3 | 77.7074 | 84.9100 | 77.7393 | 73.3708 |
| 100 | 4 | 77.5259 | 85.3417 | 78.5307 | 74.4194 |
| 200 | 1 | 78.3108 | 85.3880 | 78.7248 | **74.8743** |
| 200 | 2 | 77.8984 | 85.6346 | 78.5539 | 74.3568 |
| 200 | 3 | 78.3328 | 85.1336 | 78.1877 | 74.1486 |
| 200 | 4 | 77.7996 | 85.0054 | 78.2598 | 74.4790 |

The validation-locked winner is `hidden=100, window=1`. Relative to the in-sweep
`hidden=200, window=2` control, it changes validation by `+0.6015`, Test 8-rate by
`+0.3426`, high-missing by `+0.4575`, and miss0 by `-0.3863` percentage points.

The screen therefore finds a modest mixed-rate/high-missing improvement but does not
solve the MOSI miss0 gap. Hidden and window tuning is closed after a five-seed
confirmation of the selected `100/1` setting; no additional grid expansion is
justified.

## Final audit

- 36/36 metrics and 36/36 configs exist;
- every configuration has exactly seeds 66, 67, and 68;
- all configs use `time_attention=false` and the locked common protocol;
- all eight test rates exist in every metrics file;
- all 24 seed/rate mask hashes are identical across the 12 configurations;
- test metrics were not used to select the winner.
