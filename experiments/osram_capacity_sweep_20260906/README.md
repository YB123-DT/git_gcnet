# OSRAM capacity sweep — CMU-MOSI

Bounded capacity diagnostics on OSRAM+structured predictor. All variants use the same frozen features, cyclic schedule, masks, five seeds (66–70), optimizer, loss, and one-checkpoint 8-rate-mean Test-oracle selection. Per-rate values are additional independent Test-oracle diagnostics.

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
| H4-KV64-700 | 4 | 64 | 64 | 700 | 7,181,201 | 80.082 | +0.034 | 80.526 | -0.128 | 76.260 | -0.019 |
| H4-Output900 | 4 | 32 | 32 | 900 | 6,070,977 | 79.909 | -0.140 | 80.636 | -0.018 | 76.421 | +0.142 |
| H4-Output1000 | 4 | 32 | 32 | 1000 | 6,415,177 | 79.900 | -0.149 | 80.587 | -0.066 | 76.013 | -0.266 |

Individual folders contain design/provenance notes, compact raw JSON, and CSV summaries.
