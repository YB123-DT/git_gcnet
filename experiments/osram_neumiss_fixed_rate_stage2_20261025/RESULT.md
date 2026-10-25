# Fixed-rate frozen-parent stage-two MOSI66

Source: `0870e99179e34c448c93c9aae60688de0ecb9231`. Internal Test-oracle exploration.

NeuMiss rate 0.0 is an epoch-zero replay/no-op; the other fifteen runs train 150 epochs.

| Method | Rate/scope | ACC | WF1 | MSE | Parent WF1 | Δ ACC | Δ WF1 | Δ MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neumiss | mean | 79.6303 | 79.6122 | 1.4937 | 79.4604 | 0.0953 | 0.1518 | -0.0025 |
| neumiss | high | 73.3232 | 73.3333 | 1.8035 | 73.3516 | -0.1524 | -0.0182 | -0.0038 |
| signature | mean | 79.8209 | 79.8539 | 1.4835 | 79.4604 | 0.2858 | 0.3934 | -0.0126 |
| signature | high | 73.8313 | 73.9220 | 1.8320 | 73.3516 | 0.3557 | 0.5704 | 0.0247 |
| neumiss | 0.0 | 87.6524 | 87.6299 | 1.1520 | 87.6299 | 0.0000 | 0.0000 | 0.0000 |
| signature | 0.0 | 88.1098 | 88.0634 | 1.0147 | 87.6299 | 0.4573 | 0.4335 | -0.1373 |
| neumiss | 0.1 | 84.9085 | 84.7815 | 1.1457 | 84.7698 | 0.0000 | 0.0117 | -0.0045 |
| signature | 0.1 | 84.7561 | 84.6967 | 1.1732 | 84.7698 | -0.1524 | -0.0731 | 0.0230 |
| neumiss | 0.2 | 83.6890 | 83.7078 | 1.3671 | 82.9641 | 0.7622 | 0.7438 | -0.0073 |
| signature | 0.2 | 82.7744 | 82.8152 | 1.3645 | 82.9641 | -0.1524 | -0.1488 | -0.0099 |
| neumiss | 0.3 | 81.7073 | 81.7781 | 1.3303 | 81.7781 | 0.0000 | 0.0000 | 0.0009 |
| signature | 0.3 | 81.8598 | 81.8641 | 1.2786 | 81.7781 | 0.1524 | 0.0860 | -0.0508 |
| neumiss | 0.4 | 79.1159 | 79.0004 | 1.5439 | 78.4870 | 0.4573 | 0.5134 | 0.0024 |
| signature | 0.4 | 79.5732 | 79.6254 | 1.5413 | 78.4870 | 0.9146 | 1.1385 | -0.0002 |
| neumiss | 0.5 | 76.5244 | 76.6519 | 1.6636 | 76.8136 | -0.1524 | -0.1617 | -0.0228 |
| signature | 0.5 | 77.5915 | 77.5248 | 1.6071 | 76.8136 | 0.9146 | 0.7112 | -0.0793 |
| neumiss | 0.6 | 73.3232 | 73.4442 | 1.8154 | 73.6467 | -0.3049 | -0.2025 | 0.0088 |
| signature | 0.6 | 73.3232 | 73.4852 | 1.8778 | 73.6467 | -0.3049 | -0.1615 | 0.0712 |
| neumiss | 0.7 | 70.1220 | 69.9040 | 1.9317 | 69.5945 | 0.0000 | 0.3095 | 0.0026 |
| signature | 0.7 | 70.5793 | 70.7560 | 2.0111 | 69.5945 | 0.4573 | 1.1615 | 0.0820 |


## Scope and provenance

Remote artifacts: `/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1` on `ssh biggpu`; cmerc-repro, seed66. Results were recomputed from saved selected predictions, not newly sampled test masks. Per-rate Test-oracle selection is internal exploration, not independent generalization. One seed does not establish statistical significance; missing rates are not independent repeated seeds.

This stage2 baseline is the protected accepted per-rate Flat (mean W-F1 79.460447), NOT the new RNG-repaired joint-run Flat (79.258327). The parent is fixed and all train/test computations use the respective single rate. Additional stage2 budget and eight independent correction models must be stated. Signature high-missing W-F1 improves but its high-missing MSE increases; neither metric conflict nor negative rates are hidden.
