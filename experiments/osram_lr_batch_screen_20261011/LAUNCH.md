# Launch record

`INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle`

- Server: `biggpu`
- Final code commit: `d3ed5b6`
- Remote root: `/data2/yb/remote_experiments/osram_lr_batch_screen_20261011`
- All five seed-66 jobs were launched concurrently on 2026-10-11 UTC.
- Existing GPU2/GPU3 experiments were not changed; host GPU4 was not used.

| Run | Host GPU | PID | Epochs | Batch | Constant LR |
|---|---:|---:|---:|---:|---:|
| MOSEI Flat batch-size control | 0 | 195152 | 60 | 64 | 2e-3 |
| MOSI Flat low-LR | 7 | 199594 | 150 | 32 | 1e-4 |
| MOSI Nested low-LR | 5 | 189681 | 150 | 32 | 1e-4 |
| MOSEI Flat low-LR | 6 | 195160 | 60 | 32 | 1e-4 |
| MOSEI Nested low-LR | 6 | 189683 | 60 | 32 | 1e-4 |

The first Flat launch exposed a pre-training configuration compatibility error:
the current `TrainConfig` requires the literal `"none"` token for a disabled
meaningful block. No failed process entered training. Commit `d3ed5b6` fixes the
token, and only the three affected Flat jobs were relaunched. MOSI Flat was then
moved from GPU5 to idle GPU7 after epoch 1 because the two MOSI jobs together
occupied 32.4 GiB; the preserved first attempt is named
`gpu5_capacity_restart`. This changed placement only, not the experiment config.
