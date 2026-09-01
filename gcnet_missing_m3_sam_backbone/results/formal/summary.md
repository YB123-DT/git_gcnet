# Mask-Aware SAM Backbone — CMU-MOSI miss0

Selection: validation loss. Test was evaluated once after training.
Control: inherited `m3_mosi`; it was not retrained.

| Seed | SAM W-F1 | Control W-F1 | Delta |
|---:|---:|---:|---:|
| 66 | 84.6449% | 86.7136% | -2.0687% |
| 67 | 85.9422% | 85.9120% | +0.0302% |
| 68 | 84.5167% | 86.3082% | -1.7915% |
| 69 | 82.8531% | 86.6819% | -3.8288% |
| 70 | 84.0421% | 87.4665% | -3.4244% |

Candidate mean: 84.3998%
Control mean: 86.6165%
Mean delta: -2.2167%
Positive seeds: 1/5
Collapsed seeds: []
Gate: **FAIL**
