# Completed MOSEI Flat seeds, individually

INTERNAL TEST-ORACLE SCREEN. Read-only verification on 2026-10-24; no new
training or inference. Original Flat1600 with Local Skip; cmerc-repro,
constant LR1e-4, Adam/WD1e-5, batch32,60epochs, MSE,
gcnet-random-mask-v1 cyclic rates0.0-0.7. Per-rate TEST-best W-F1; ACC and MSE
are from the same selected predictions. High means rates0.5/0.6/0.7.

| Seed | Eight-rate ACC | Eight-rate W-F1 | High W-F1 |
|---:|---:|---:|---:|
| 66 | 83.492708 | 83.288796 | 80.035750 |
| 67 | 83.396395 | 83.248318 | 80.137004 |
| 68 | 83.468630 | 83.305343 | 79.962090 |
| Three-seed mean | 83.452578 | 83.280819 | 80.044948 |

Remote root:
`/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1`.
Runs: `runs/CMUMOSEI/seed_{66,67,68}/flat`.
Runner SHA: `b0a712ed42085477fea986fbd8cd7bfc6a803e53`.
Numerical SHA: `0da38b3326fdad09da0195e49b8b7d6424b4f613`.

Verified complete60-epoch histories, configuration, selected prediction SHA256,
same-call selected metrics and aggregate arithmetic for all three seeds.
SUMMARY.json SHA256:

- 66: `e4508e51dcab802c859d33a8f9bbba376b11414837c550bfc2b456be053c64b6`
- 67: `aa0de5bfd87463ff0d7cc67126d91d4b63958aa838705a8672c96837b5cac9e2`
- 68: `e80b20ef9aeddbc3f8b6f8a9309ce5f06ff235c59c6a8bf2aab26c25b479d48b`

Do not substitute the later Nested-screen Flat seed66 W-F1=83.477635 into this
three-seed mean. That is a separate same-source matched-control training run
with the explicit Torch1.13 UUID CUDA RNG repair. This earlier runner does not
invoke that repair. Same environment/LR/missing protocol alone does not make
the two training executions identical. Neither result is deleted or relabeled.
