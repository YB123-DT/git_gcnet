# MOSEI Nested combinations

INTERNAL TEST-ORACLE SCREEN; seed66 only.
Same cmerc-repro, fixed accepted masks,60epochs,constant1e-4,batch32,WD1e-5,MSE.
Controls are reused read-only. Numerical sources identical except verified registry-only additions.
No independent generalization or statistical-significance claim. No automatic promotion.

| Method | ACC | ΔACC pp | W-F1 | ΔWF1 pp | High WF1 | Nested params |
|---|---:|---:|---:|---:|---:|---:|
| none | 83.627 | +0.000 | 83.478 | +0.000 | 79.898 | 0 |
| nested_dim32 | 83.665 | +0.038 | 83.549 | +0.071 | 80.133 | 65667 |
| mosei_nested_mlp128 | 83.675 | +0.048 | 83.481 | +0.003 | 80.117 | 216899 |
| mosei_nested_d32_l3_g8_m128 | 83.431 | -0.196 | 83.239 | -0.239 | 79.870 | 109155 |
| mosei_nested_d32_l3_g8_m64 | 83.376 | -0.251 | 83.194 | -0.284 | 79.930 | 80163 |
| mosei_nested_d32_l2_g8_m128 | 83.658 | +0.031 | 83.441 | -0.037 | 80.180 | 96642 |
| mosei_nested_d32_l3_g4_m128 | 83.627 | -0.000 | 83.498 | +0.020 | 80.119 | 108899 |
