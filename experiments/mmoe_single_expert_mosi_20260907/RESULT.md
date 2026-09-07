# MMoE routing weak-point diagnostic

**Internal diagnostic only; not a formal paper result.**

Full-4E is inherited from the H8/32/32/700 OSRAM run. Single-1E was
trained with the same cyclic mixed-rate protocol and changes only
`num_experts=1, top_k=1`. The table uses the same per-rate Test-oracle
extraction convention as the current OSRAM diagnostic; each seed's
eight-rate Test-oracle checkpoint mean is also recorded.

| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |
|---|---:|---:|
| Full-4E | 80.781% | 76.586% |
| Single-1E | 80.851% | 76.677% |

## Strict one-checkpoint cross-rate check

The table above intentionally gives each rate its own best epoch,
matching the requested per-rate diagnostic. As a guard against that
optimistic view, the same histories were also scored with one
eight-rate-mean Test-oracle checkpoint per seed:

| Condition | mean over 5 one-checkpoint scores | selected epochs (66,67,68,69,70) |
|---|---:|---|
| Full-4E | 80.267% | [61, 39, 46, 38, 38] |
| Single-1E | 80.267% | [73, 43, 55, 44, 42] |
| Δ | +0.001 percentage points | — |

| Rate | Full-4E | Single-1E | Δ | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.639% ± 0.669 | 87.340% ± 0.184 | -0.300 | 2/5 |
| 0.1 | 85.365% ± 0.357 | 85.418% ± 0.761 | +0.053 | 3/5 |
| 0.2 | 82.693% ± 1.090 | 82.877% ± 1.436 | +0.184 | 4/5 |
| 0.3 | 81.439% ± 0.651 | 81.452% ± 1.118 | +0.014 | 2/5 |
| 0.4 | 79.350% ± 1.521 | 79.691% ± 1.675 | +0.340 | 4/5 |
| 0.5 | 77.300% ± 0.825 | 77.455% ± 1.129 | +0.155 | 3/5 |
| 0.6 | 76.804% ± 0.535 | 76.455% ± 0.550 | -0.348 | 0/5 |
| 0.7 | 75.654% ± 1.940 | 76.119% ± 1.869 | +0.465 | 4/5 |

## Per-seed eight-rate means

| Seed | Full-4E | Single-1E | Δ |
|---:|---:|---:|---:|
| 66 | 80.839% | 81.024% | +0.185 |
| 67 | 80.414% | 80.476% | +0.061 |
| 68 | 81.094% | 80.988% | -0.106 |
| 69 | 80.981% | 80.872% | -0.109 |
| 70 | 80.575% | 80.895% | +0.321 |

## Diagnosis

The single-expert control is a mechanism screen rather than a
parameter-matched final model. A drop indicates that expert
specialization/routing contributes useful capacity; parity or an
improvement indicates that routing is a likely weak point and that a
later parameter-matched shared predictor would be warranted.

## Routing diagnostics

The final training histories show that the Full-4E condition uses
nonzero expert specialization, while Single-1E has zero routing
entropy by construction.  This is evidence about the mechanism, not
a claim that the lower-capacity control is a fair final model; the
parameter counts and averaged routing statistics are in
`diagnostics.json`.
