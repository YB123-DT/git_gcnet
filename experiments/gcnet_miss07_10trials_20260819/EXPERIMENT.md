# GCNet Session-5 missing=0.7 repeated trials

Ten complete training trials were run on biggpu with seeds 66--75. The model was
the GCNet path with the modality predictor and JEPA loss disabled. GPU4 was not
used. The requested missing rate of 0.7 realizes exactly 2/3 missing because the
mask implementation retains one of three modalities per utterance.

| Seed | W-F1 (%) | Best epoch |
|---:|---:|---:|
| 66 | 59.49 | 98 |
| 67 | 64.56 | 98 |
| 68 | 61.35 | 93 |
| 69 | 61.29 | 70 |
| 70 | 58.91 | 88 |
| 71 | 62.01 | 98 |
| 72 | 59.13 | 65 |
| 73 | 63.52 | 89 |
| 74 | 56.17 | 100 |
| 75 | 59.82 | 80 |

Mean W-F1 is **60.63%**, sample standard deviation is **2.44 points**, and the
95% t-interval is **60.63 +/- 1.75 points**. The range is 56.17--64.56. Thus the
old single-run 64.25 is a plausible high-tail trial, not the expected score.

The loader optimization was benchmarked on the same remote dataset. A first
indexed scan and cache build took 10.67 seconds; loading the resulting 234 MB
cache took 0.10 seconds. Previously each process spent roughly 5--8 minutes
performing repeated glob scans. Future runs default to six Torch CPU threads per
process to avoid oversubscribing the 80-core host.

Machine-readable results are in `summary.json`.
