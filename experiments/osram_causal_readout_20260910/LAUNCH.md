# Launch evidence

Code commit: `9d795f0`, branch `feature/osram-complete`.

All 15 processes launched on biggpu; launcher PID2036093, child PIDs2036094–2036108. `LAUNCH_SNAPSHOT.json` is a point-in-time copy, not a live status file. At first inspection all children were alive, all loaded the expected 512/1024/1024 feature dimensions, and initial epoch training/test evaluation lines were present. No new selected F1 result is claimed.

| GPU | Seed | Concurrent variants |
|---|---|---|
| 2 | 66 | Local-only, Local+Base, Local+Gap |
| 3 | 67 | Local-only, Local+Base, Local+Gap |
| 5 | 68 | Local-only, Local+Base, Local+Gap |
| 6 | 69 | Local-only, Local+Base, Local+Gap |
| 7 | 70 | Local-only, Local+Base, Local+Gap |

Live queue/logs: `/data2/yb/remote_experiments/osram_causal_readout_20260910/`.
After all processes exit, the launcher automatically builds the per-rate Test-oracle summary. No further models or datasets are queued. Full is inherited; no Full training process was launched.
