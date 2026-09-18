# MOSI causal OSRAM hyperparameter screening

Internal diagnostic only; not a formal paper result.

Each configuration is seed 66, cyclic missing-rate training, and independent Test-oracle epoch selection for every rate.

| rank | config | group | all-8 mean | high-missing mean | epochs | latent | output | heads | key/value |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | cfg01_baseline | optimization | 80.828 | 76.088 | 100 | 256 | 700 | 8 | 32/32 |
| 2 | cfg11_capacity_baseline | capacity | 80.828 | 76.088 | 100 | 256 | 700 | 8 | 32/32 |
| 3 | cfg56_adam_lr1e3_constant400 | schedule | 80.828 | 76.088 | 400 | 256 | 700 | 8 | 32/32 |
| 4 | cfg35_lr1e3_b32_d0 | optimization | 80.662 | 75.782 | 100 | 256 | 700 | 8 | 32/32 |
| 5 | cfg17_heads4_kv64 | capacity | 80.608 | 75.855 | 100 | 256 | 700 | 4 | 64/64 |
| 6 | cfg23_adamw_lr1e3_wd1e3_d3 | schedule | 80.602 | 75.584 | 100 | 256 | 700 | 8 | 32/32 |
| 7 | cfg18_out1024_kv48 | capacity | 80.586 | 76.202 | 100 | 256 | 1024 | 8 | 48/48 |
| 8 | cfg29_long200_clip05 | schedule | 80.560 | 75.738 | 200 | 256 | 700 | 8 | 32/32 |
| 9 | cfg47_lat512_out1400_kv32 | capacity | 80.548 | 75.580 | 100 | 512 | 1400 | 8 | 32/32 |
| 10 | cfg07_lr3e4_b8_d1 | optimization | 80.512 | 75.561 | 100 | 256 | 700 | 8 | 32/32 |
| 11 | cfg12_out1400 | capacity | 80.477 | 75.901 | 100 | 256 | 1400 | 8 | 32/32 |
| 12 | cfg15_lat512_out1024_kv64 | capacity | 80.475 | 75.692 | 100 | 512 | 1024 | 8 | 64/64 |
| 13 | cfg26_adam_cosine_lr3e4 | schedule | 80.468 | 75.500 | 100 | 256 | 700 | 8 | 32/32 |
| 14 | cfg42_out900_kv32 | capacity | 80.439 | 75.478 | 100 | 256 | 900 | 8 | 32/32 |
| 15 | cfg08_lr1e3_b8_d1 | optimization | 80.371 | 75.732 | 100 | 256 | 700 | 8 | 32/32 |
| 16 | cfg53_adamw_lr3e4_cosine200 | schedule | 80.370 | 75.824 | 200 | 256 | 700 | 8 | 32/32 |
| 17 | cfg49_heads8_out1024_kv64 | capacity | 80.368 | 75.890 | 100 | 256 | 1024 | 8 | 64/64 |
| 18 | cfg34_lr3e4_b32_d0 | optimization | 80.337 | 75.512 | 100 | 256 | 700 | 8 | 32/32 |
| 19 | cfg21_adamw_lr1e3_wd1e4_d3 | schedule | 80.334 | 75.337 | 100 | 256 | 700 | 8 | 32/32 |
| 20 | cfg48_heads4_out1400_kv32 | capacity | 80.307 | 75.472 | 100 | 256 | 1400 | 4 | 32/32 |
| 21 | cfg19_lat384_out1024_kv48 | capacity | 80.305 | 75.612 | 100 | 384 | 1024 | 8 | 48/48 |
| 22 | cfg20_out1400_kv48 | capacity | 80.285 | 75.633 | 100 | 256 | 1400 | 8 | 48/48 |
| 23 | cfg44_out1400_kv48 | capacity | 80.285 | 75.633 | 100 | 256 | 1400 | 8 | 48/48 |
| 24 | cfg40_lr1e3_proj2_bb05_cls1p5 | optimization | 80.250 | 75.295 | 100 | 256 | 700 | 8 | 32/32 |
| 25 | cfg16_heads16_kv32 | capacity | 80.222 | 75.252 | 100 | 256 | 700 | 16 | 32/32 |
| 26 | cfg02_lr3e4 | optimization | 80.211 | 75.206 | 100 | 256 | 700 | 8 | 32/32 |
| 27 | cfg54_adam_lr3e4_constant200 | schedule | 80.211 | 75.206 | 200 | 256 | 700 | 8 | 32/32 |
| 28 | cfg14_out1400_kv64 | capacity | 80.179 | 75.331 | 100 | 256 | 1400 | 8 | 64/64 |
| 29 | cfg28_adamw_cosine_lr3e4 | schedule | 80.154 | 74.856 | 100 | 256 | 700 | 8 | 32/32 |
| 30 | cfg50_heads16_out1024_kv64 | capacity | 80.139 | 75.362 | 100 | 256 | 1024 | 16 | 64/64 |
| 31 | cfg46_lat384_out1400_kv64 | capacity | 80.138 | 75.599 | 100 | 384 | 1400 | 8 | 64/64 |
| 32 | cfg51_adamw_lr1e4_cosine200 | schedule | 80.127 | 75.428 | 200 | 256 | 700 | 8 | 32/32 |
| 33 | cfg57_adamw_lr1e4_cosine400 | schedule | 80.121 | 75.101 | 400 | 256 | 700 | 8 | 32/32 |
| 34 | cfg30_long400_adamw_cosine | schedule | 80.118 | 75.205 | 400 | 256 | 700 | 8 | 32/32 |
| 35 | cfg43_out1100_kv32 | capacity | 80.070 | 75.630 | 100 | 256 | 1100 | 8 | 32/32 |
| 36 | cfg24_adamw_lr3e4_wd1e3_d2 | schedule | 80.065 | 75.079 | 100 | 256 | 700 | 8 | 32/32 |
| 37 | cfg39_lr3e4_wd1e2_d3 | optimization | 80.060 | 75.222 | 100 | 256 | 700 | 8 | 32/32 |
| 38 | cfg10_lr1e3_wd1e4_d3_proj0_bb05 | optimization | 80.056 | 75.344 | 100 | 256 | 700 | 8 | 32/32 |
| 39 | cfg52_adam_lr1e4_cosine200 | schedule | 80.045 | 75.241 | 200 | 256 | 700 | 8 | 32/32 |
| 40 | cfg45_lat320_out900_kv40 | capacity | 79.980 | 75.316 | 100 | 320 | 900 | 8 | 40/40 |
| 41 | cfg05_lr3e4_b16_d2 | optimization | 79.962 | 74.691 | 100 | 256 | 700 | 8 | 32/32 |
| 42 | cfg09_lr3e4_wd1e3_d3 | optimization | 79.913 | 75.176 | 100 | 256 | 700 | 8 | 32/32 |
| 43 | cfg38_lr3e4_wd1e6_d3 | optimization | 79.907 | 74.929 | 100 | 256 | 700 | 8 | 32/32 |
| 44 | cfg55_adamw_lr1e3_constant200 | schedule | 79.897 | 75.137 | 200 | 256 | 700 | 8 | 32/32 |
| 45 | cfg13_kv64 | capacity | 79.862 | 75.457 | 100 | 256 | 700 | 8 | 64/64 |
| 46 | cfg22_adamw_lr3e4_wd1e4_d2 | schedule | 79.825 | 75.178 | 100 | 256 | 700 | 8 | 32/32 |
| 47 | cfg37_lr3e4_wd0_d3 | optimization | 79.823 | 74.993 | 100 | 256 | 700 | 8 | 32/32 |
| 48 | cfg06_lr1e3_b16_d2 | optimization | 79.751 | 75.079 | 100 | 256 | 700 | 8 | 32/32 |
| 49 | cfg41_out500_kv32 | capacity | 79.641 | 74.690 | 100 | 256 | 500 | 8 | 32/32 |
| 50 | cfg04_lr1e4_b16_d3 | optimization | 79.599 | 74.697 | 100 | 256 | 700 | 8 | 32/32 |
| 51 | cfg60_adam_lr1e3_cosine_clip5_cls05 | schedule | 79.585 | 73.985 | 100 | 256 | 700 | 8 | 32/32 |
| 52 | cfg59_adamw_lr3e4_cosine_clip05 | schedule | 79.469 | 74.411 | 100 | 256 | 700 | 8 | 32/32 |
| 53 | cfg27_adamw_cosine_lr1e3 | schedule | 79.350 | 73.997 | 100 | 256 | 700 | 8 | 32/32 |
| 54 | cfg31_lr3e5_b32_d5 | optimization | 79.297 | 74.895 | 100 | 256 | 700 | 8 | 32/32 |
| 55 | cfg33_lr1e4_b32_d1 | optimization | 79.238 | 74.397 | 100 | 256 | 700 | 8 | 32/32 |
| 56 | cfg25_adam_cosine_lr1e3 | schedule | 79.206 | 73.938 | 100 | 256 | 700 | 8 | 32/32 |
| 57 | cfg03_lr3e3 | optimization | 79.179 | 74.878 | 100 | 256 | 700 | 8 | 32/32 |
| 58 | cfg36_lr3e3_b16_d5 | optimization | 79.070 | 74.121 | 100 | 256 | 700 | 8 | 32/32 |
| 59 | cfg32_lr3e5_b16_d3 | optimization | 78.787 | 73.541 | 100 | 256 | 700 | 8 | 32/32 |
| 60 | cfg58_adam_lr3e3_clip05 | schedule | 58.468 | 58.530 | 100 | 256 | 700 | 8 | 32/32 |
