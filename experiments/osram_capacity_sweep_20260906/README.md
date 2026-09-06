# OSRAM capacity sweep — CMU-MOSI

This directory records a bounded capacity sweep on OSRAM+structured predictor. All variants use the same frozen features, cyclic rate schedule, masks, five seeds (66–70), optimizer, loss, and one-checkpoint 8-rate-mean Test-oracle selection. The per-rate columns are additional independent Test-oracle diagnostics.

**Internal diagnostic only; not a formal paper result.**

H4 control (`heads=4,key/value=32,output=500`) from `experiments/osram_complete_20260906/`: single-checkpoint mean **80.048%**; per-rate mean **80.654%**; high-missing mean **76.279%**.

| Variant | Heads | Key | Value | Output | Params | 8-rate selected mean | Δ vs H4 | Per-rate oracle mean | Δ vs H4 | High-missing mean | Δ vs H4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H8 | 8 | 32 | 32 | 500 | 6,428,017 | 79.987 | -0.061 | 80.616 | -0.038 | 76.443 | +0.164 |
| KV64 | 4 | 64 | 64 | 500 | 6,428,001 | 79.690 | -0.359 | 80.372 | -0.281 | 76.042 | -0.237 |
| V64 | 4 | 32 | 64 | 500 | 5,903,201 | 80.070 | +0.022 | 80.715 | +0.061 | 76.343 | +0.064 |
| Out700 | 4 | 32 | 32 | 700 | 5,442,577 | 79.823 | -0.225 | 80.556 | -0.097 | 76.195 | -0.083 |
| H8+Out700 | 8 | 32 | 32 | 700 | 7,181,217 | 80.267 | +0.219 | 80.781 | +0.127 | 76.586 | +0.307 |
| H8+V64+Out700 | 8 | 32 | 64 | 700 | 9,608,865 | 80.124 | +0.076 | 80.702 | +0.048 | 76.441 | +0.162 |

Each variant folder contains design/provenance notes, compact raw JSON, and machine-readable summaries. The H8+V64+Output700 follow-up used one retry for seed 69 after a transient CUDA initialization failure.
