# Paired warmup/cosine screen

INTERNAL DIAGNOSTIC ONLY

MOSI seed66. Original Flat and original Nested, sealed source ad211c0. Two new100epoch runs; reuse completed constant controls. Per-rate Test-oracle BEST, not independent paper performance. Cyclic random missing, original masks/loss/model/head/batch protocol unchanged. Added ONLY existing warmup/cosine package:5epochs warmup, peak1e-3, then cosine to0 at100. This does not isolate pure cosine from warmup or alternative schedules.

| Model | Constant mean8 | Cosine mean8 | Delta pp | Constant high | Cosine high | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| flat | 81.068 | 80.029 | -1.039 | 76.352 | 75.211 | -1.141 |
| nested | 80.992 | 80.323 | -0.670 | 76.077 | 75.974 | -0.103 |

| Rate | Flat constant | Flat cosine | Nested constant | Nested cosine |
|---|---:|---:|---:|---:|
| 0.0 | 88.205 | 87.218 | 88.078 | 86.962 |
| 0.1 | 86.507 | 85.475 | 86.358 | 85.300 |
| 0.2 | 83.187 | 82.142 | 83.735 | 82.177 |
| 0.3 | 80.763 | 80.386 | 80.523 | 80.310 |
| 0.4 | 80.827 | 79.377 | 81.012 | 79.909 |
| 0.5 | 77.494 | 76.063 | 77.675 | 76.460 |
| 0.6 | 75.790 | 74.278 | 75.032 | 75.759 |
| 0.7 | 75.773 | 75.293 | 75.525 | 75.704 |

Nested-minus-Flat mean8 gap: constant-0.076pp, cosine+0.293pp. Relative comparison improves, but BOTH cosine models score below their own constant version; do not call this an absolute Nested improvement.

All four runs complete, all20artifact hashes/run and gradient hashes verified;200paired training masks identical across models/schedules. All200cosine learning-rate file hashes and actual group rates match the full expected curve, finalLR0. per_rate.csv includes all BEST epochs; separate CSVs preserve actualLR traces. Full recovery and eight BEST checkpoints retained on biggpu.

Conclusion: this seed66 warmup/cosine configuration does not improve final BEST W-F1. Earlier convergence is not final performance gain. No automatic further seeds, schedule search or structural modifications. No inference or training added for this report. Single seed: no statistical significance or multi-seed consistency claim.
