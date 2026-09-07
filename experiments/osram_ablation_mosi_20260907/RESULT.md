# OSRAM mechanism ablation result

**Internal diagnostic only; not a formal paper result.**

Each seed is one cyclic mixed-rate training run.  For the table below,
each missing rate selects its own best Test weighted-F1 epoch from that
history (`per-rate-test-oracle`).  Full OSRAM is inherited from the
H8/32/32/700 run in `experiments/osram_heads8_out700_20260906/`.

| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |
|---|---:|---:|
| Full OSRAM | 80.781% | 76.586% |
| Local+Base | 80.305% | 75.851% |
| Local-only | 77.950% | 72.187% |

| Rate | Full OSRAM | Local+Base | Δ | Local-only | Δ |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.639% ± 0.669 | 87.358% | -0.281 | 87.152% | -0.488 |
| 0.1 | 85.365% ± 0.357 | 84.893% | -0.472 | 84.147% | -1.218 |
| 0.2 | 82.693% ± 1.090 | 82.528% | -0.165 | 81.075% | -1.618 |
| 0.3 | 81.439% ± 0.651 | 81.387% | -0.051 | 78.929% | -2.510 |
| 0.4 | 79.350% ± 1.521 | 78.719% | -0.631 | 75.735% | -3.615 |
| 0.5 | 77.300% ± 0.825 | 76.622% | -0.679 | 73.079% | -4.221 |
| 0.6 | 76.804% ± 0.535 | 76.005% | -0.798 | 72.619% | -4.184 |
| 0.7 | 75.654% ± 1.940 | 74.927% | -0.727 | 70.862% | -4.792 |

## Interpretation

Local-only removes both memory readout families and is the direct
current-utterance control.  Local+Base keeps ordinary bidirectional
memory but masks the three missing-specific gap slots.  The Full minus
Local+Base difference therefore estimates the incremental contribution
of the gap-conditioned readout under this diagnostic protocol.

The ablation switch does not change the OSRAM scan or parameter count;
it masks only the selected context slots before the emotion head and
structured predictor.  Results are optimistic Test-oracle diagnostics
and must not be presented as validation-selected benchmark scores.
