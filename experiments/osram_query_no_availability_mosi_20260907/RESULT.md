# OSRAM Query availability diagnostic

**Internal diagnostic only; not a formal paper result.**

The only newly trained condition removes explicit `a_t` from the Query.
Key/Value conditioning, hard gap selection, memory write, predictor and
all training settings remain unchanged.

| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |
|---|---:|---:|
| Explicit `a_t` | 80.781% | 76.586% |
| No explicit `a_t` | 80.208% | 75.840% |

## Strict one-checkpoint cross-rate check

| Condition | mean | selected epochs (66,67,68,69,70) |
|---|---:|---|
| Explicit `a_t` | 80.267% | [61, 39, 46, 38, 38] |
| No explicit `a_t` | 79.641% | [68, 60, 73, 42, 53] |
| Δ (no explicit − explicit) | -0.626 percentage points | — |

| Seed | Explicit `a_t` | No explicit `a_t` | Δ |
|---:|---:|---:|---:|
| 66 | 80.229% | 79.823% | -0.405 |
| 67 | 79.850% | 79.379% | -0.472 |
| 68 | 80.880% | 79.553% | -1.327 |
| 69 | 80.498% | 79.333% | -1.165 |
| 70 | 79.877% | 80.117% | +0.240 |

| Rate | Explicit `a_t` | No explicit `a_t` | Δ | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.639% ± 0.669 | 87.331% ± 0.485 | -0.309 | 3/5 |
| 0.1 | 85.365% ± 0.357 | 84.807% ± 0.723 | -0.558 | 1/5 |
| 0.2 | 82.693% ± 1.090 | 82.271% ± 0.749 | -0.422 | 1/5 |
| 0.3 | 81.439% ± 0.651 | 81.128% ± 0.356 | -0.310 | 1/5 |
| 0.4 | 79.350% ± 1.521 | 78.605% ± 1.796 | -0.745 | 1/5 |
| 0.5 | 77.300% ± 0.825 | 76.525% ± 1.304 | -0.776 | 2/5 |
| 0.6 | 76.804% ± 0.535 | 76.074% ± 0.912 | -0.730 | 1/5 |
| 0.7 | 75.654% ± 1.940 | 74.920% ± 1.687 | -0.734 | 0/5 |

The per-rate rows use the requested diagnostic convention in which each
rate is allowed to select its own Test-oracle epoch. The strict table is
the guard using one eight-rate-mean Test-oracle checkpoint per seed.

## Decision

Explicit `a_t` is retained. Removing it lowers the per-rate eight-rate
mean by 0.573 percentage points and the high-missing mean by 0.746
points; the strict one-checkpoint guard also drops by 0.626 points.
This ablation does not support removing availability from the Query.
