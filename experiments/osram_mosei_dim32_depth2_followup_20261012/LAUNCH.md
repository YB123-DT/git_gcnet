# Launch record

`INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle`

- Server: `biggpu`
- Code commit: `e1a0862`
- Remote root: `/data2/yb/remote_experiments/osram_mosei_dim32_depth2_followup_20261012`
- Started: 2026-10-12 00:37 UTC
- Existing seed66 / LR 1e-3 result is reused; its selected epochs are all below
  60, so the seed67/68 follow-ups use the standing 60-epoch MOSEI budget.

| Run | GPU | PID | Epochs | LR |
|---|---:|---:|---:|---:|
| seed67 dim32+depth2 | 5 | 2734820 | 60 | 1e-3 |
| seed68 dim32+depth2 | 5 | 2734823 | 60 | 1e-3 |
| seed66 dim32+depth2 low-LR | 6 | 2713048 | 60 | 1e-4 |

All three processes were confirmed alive with GPU allocations. Host GPU4 was
not used. Outputs are isolated under `execution/runs/`.

The initial seed67/68 processes configured for 100 epochs were stopped before
finishing epoch 1 and preserved with the suffix `stopped_before_epoch1`. They
were replaced by the 60-epoch runs above; seed66 low-LR was not interrupted.

## Additional seed68 variants

| Run | GPU | PID | Epochs | LR |
|---|---:|---:|---:|---:|
| seed68 MLP128 | 5 | 3318442 | 60 | 1e-3 |
| seed68 depth2 | 5 | 3321634 | 60 | 1e-3 |

Depth2 was initially placed on GPU6, but another process claimed 23.7 GiB after
the resource check. It was moved before completing epoch 1 to GPU5; the
preserved initial directory uses suffix `gpu6_capacity_restart`. The experiment
configuration was unchanged.
