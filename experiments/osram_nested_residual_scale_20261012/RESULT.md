# Original Nested correction strength — one slot at a time

**INTERNAL DIAGNOSTIC ONLY.** These are frozen Test-oracle checkpoint
interventions, not deployable model scores and not a newly trained method.

## Protocol

- MOSI seeds 66/67/68, original `nested_gnn_rooted_evidence` per-rate BEST
  checkpoints and original eight cyclic test masks.
- No training and no parameter update. Memory is scanned once per batch.
- For one slot at a time, apply
  `raw + gamma * (nested_corrected - raw)`, with
  `gamma in {0, 0.5, 1, 1.5, 2}`. The other four slots remain exactly at 1.
- Slots are Local, Base, Gap-A, Gap-T and Gap-V. The original Local Skip still
  receives the unmodified Local; inactive Gaps remain masked in Flat.
- `gamma=1` exactly replays the original Nested model. The best coefficient in
  each row below is selected on the same test diagnostic and must not be read as
  a validation-selected hyperparameter.

## Three-seed results

W-F1 (%); each cell is the equal mean over three seeds and eight rates.

| Slot | gamma=0 | gamma=.5 | gamma=1 | gamma=1.5 | gamma=2 | Best delta vs 1 |
|---|---:|---:|---:|---:|---:|---:|
| Local | **80.494448** | 80.494093 | 80.488818 | 80.488818 | 80.489938 | +0.005630 |
| Base | 80.468488 | 80.485117 | 80.488818 | **80.498193** | 80.478228 | +0.009375 |
| Gap-A | 80.469150 | 80.476342 | 80.488818 | 80.488818 | **80.502145** | +0.013327 |
| Gap-T | 80.467160 | 80.473106 | 80.488818 | 80.480555 | **80.489241** | +0.000423 |
| Gap-V | 80.491223 | 80.483950 | 80.488818 | 80.494249 | **80.500292** | +0.011474 |

High-missing mean (.5/.6/.7):

| Slot | gamma=0 | gamma=.5 | gamma=1 | gamma=1.5 | gamma=2 |
|---|---:|---:|---:|---:|---:|
| Local | **75.266831** | 75.265884 | 75.251816 | 75.251816 | 75.254803 |
| Base | 75.191687 | 75.204389 | 75.251816 | **75.280574** | 75.246033 |
| Gap-A | 75.199368 | 75.218547 | 75.251816 | 75.251816 | **75.287356** |
| Gap-T | 75.191501 | 75.190885 | 75.251816 | 75.249851 | **75.274318** |
| Gap-V | 75.221561 | 75.203474 | 75.251816 | 75.249967 | **75.266083** |

The per-seed mean8 values for each row's mean8-best coefficient are:

| Slot / coefficient | Seed66 | Seed67 | Seed68 | Mean |
|---|---:|---:|---:|---:|
| Original (`gamma=1`) | 80.992147 | 80.850540 | 79.623766 | 80.488818 |
| Local / 0 | 81.030615 | 80.828963 | 79.623766 | 80.494448 |
| Base / 1.5 | 81.002188 | 80.868623 | 79.623766 | 80.498193 |
| Gap-A / 2 | 81.010597 | 80.872072 | 79.623766 | 80.502145 |
| Gap-T / 2 | 80.989971 | 80.853986 | 79.623766 | 80.489241 |
| Gap-V / 2 | 81.026571 | 80.850540 | 79.623766 | 80.500292 |

Raw values are authoritative in `results/macro.csv` and
`results/per_rate.csv`; displayed per-seed values above are rounded.

## Interpretation

Separate fixed scaling does not reveal a practically meaningful strength
problem. The largest test-selected mean8 change is only +0.013327 percentage
points (Gap-A x2). Gap-A x2 produces only two wrong-to-right flips and zero
right-to-wrong flips across all 24 repeated seed/rate evaluations; seed68 is
essentially unchanged. Several coefficients trade gains at one rate for losses
at another. Therefore this result does **not** justify a learned scalar Gate or
claim that the original residual is systematically under-scaled.

## Correctness and provenance

- Code commit: `ba23176`.
- Server: `biggpu`; physical GPU6,
  `GPU-e4cafb17-818e-216a-b94a-7440063a9153`; GPU4 was not used.
- 3 focused tests passed in the sealed biggpu environment.
- 24/24 seed/rate checkpoint replays completed; 600 intervention metric rows.
- Maximum original-forward/replay and archived-prediction errors were below
  `1e-5`; checkpoint and model-state hashes remained unchanged.
- Run directory:
  `/data2/yb/remote_experiments/osram_nested_residual_scale_20261012/results`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -m \
  experiments.osram_nested_residual_scale_20261012.run \
  --output /data2/yb/remote_experiments/osram_nested_residual_scale_20261012/results \
  --gpu 6 --code-commit ba23176
```
