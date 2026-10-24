# Why two seed66 mean ACC values are equal

Read-only comparison of completed saved predictions,2026-10-24.
INTERNAL TEST-ORACLE SCREEN; no new inference or training for this check.

| Run | LR | Eight-rate ACC | Eight-rate W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| s0 original | 1e-3 | 83.492708 | 83.279076 | 80.181722 |
| s0 lower LR | 1e-4 | 83.509906 | 83.364978 | 80.108311 |
| cmerc historical three-seed batch,seed66 | 1e-4 | 83.492708 | 83.288796 | 80.035750 |

The first and third rows have equal aggregate ACC, not equal predictions or
equal selected epochs. Every rate replays identical labels/availability.
At miss0.7,368 of3634 nonneutral utterances have different predicted polarity;
mean absolute scalar prediction difference is0.240083. Comparing the two
LR1e-4 rows instead,390 of3634 differ at0.7; mean absolute difference0.295973.
Equality of aggregate accuracy does not imply copied results. Because the
first/third rows change BOTH environment and LR, their comparison cannot
isolate an environment effect.

Remote run paths under `/data2/yb/remote_experiments`:

- `osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat`
- `osram_mosei_flat_lr1e4_seed66_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat`
- `osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat`

Compared each rate's `best_by_rate.predictions` arrays directly; no random
re-evaluation. s0 is Torch2.2.2+cu121/NumPy1.24.3; cmerc is
Torch1.13.1+cu117/NumPy1.26.4. The later RNG-repaired Nested-screen Flat is a
fourth distinct run and is not one of the table's entries.
