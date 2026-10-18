# Current MELD baseline: persistent unimodal evaluation

INTERNAL DIAGNOSTIC ONLY. Evaluation-only; no training or checkpoint reselection.
Every conversation keeps only A, T(Text), or V throughout, not random-mask subgroups.
Accepted Flat1600 + Local Skip, lr1e-5 / wd1e-5, seeds66/67/68, original cmerc-repro runtime.
Each seed uses its preserved random-miss0.7 ACC-selected BEST for all three patterns.

| Seed | Persistent visible | ACC | UA | W-F1 | CE |
|---:|---|---:|---:|---:|---:|
| 66 | A | 48.084291 | 20.791067 | 40.103649 | 1.468459 |
| 66 | T | 66.321839 | 48.990948 | 65.511618 | 1.076721 |
| 66 | V | 48.084291 | 14.322666 | 31.393295 | 1.516715 |
| 67 | A | 49.272031 | 19.402017 | 39.794403 | 1.477253 |
| 67 | T | 65.862069 | 48.447455 | 64.998226 | 1.063973 |
| 67 | V | 47.356322 | 14.976415 | 32.999303 | 1.517622 |
| 68 | A | 48.084291 | 18.904247 | 39.219093 | 1.482758 |
| 68 | T | 66.245211 | 46.000375 | 64.951790 | 1.076935 |
| 68 | V | 48.084291 | 14.395153 | 31.588049 | 1.528126 |

| Pattern | mean ACC | mean UA | mean W-F1 | mean CE |
|---|---:|---:|---:|---:|
| A | 48.480204 | 19.699110 | 39.705715 | 1.476156 |
| T | 66.143040 | 47.812926 | 65.153878 | 1.072543 |
| V | 47.841635 | 14.564745 | 31.993549 | 1.520821 |

UA is seven-class macro recall; class recalls/support and sample standard deviations are in RESULTS.json.
This is a new fixed-availability intervention, not a change to training GCNet missing or to the accepted random baseline.
No weights, thresholds or hyperparameters selected using these new pattern scores.
Original selected-mask replay passes before each seed; all source checkpoint/prediction hashes stay unchanged.
No single-modality models were trained; Memory contains only the currently evaluated modality throughout the conversation.
