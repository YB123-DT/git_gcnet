# INTERNAL DIAGNOSTIC ONLY — round two interim results

Four seed66 runs finished 100 epochs with exit code 0 and complete provenance. Values below were checked against final `metrics.json` / `selected_weighted_f1_by_rate`, not an unfinished epoch. Selection remains per-rate TEST-oracle, not validation-selected paper results.

| Method | 8-rate W-F1 (%) | High-missing W-F1 (%) | Mean8 delta vs Flat (pp) |
|---|---:|---:|---:|
| Matched Flat seed66 | 81.068 | 76.352 | — |
| Dense Co-Attention | 80.286 | 75.566 | -0.782 |
| MAC | 80.171 | 75.383 | -0.897 |
| SPDNet | 80.024 | 75.227 | -1.044 |
| MCAN | 79.960 | 75.272 | -1.108 |

No completed candidate exceeds Flat. No additional seeds are promoted from these results. High-missing means rates .5/.6/.7; overall means rates .0 through .7 equally weighted.

The next authorized candidates are NLM, FSPool and RAT-SPN on physical biggpu GPUs 5, 6 and 7. Each uses the same immutable source `98268f4`, seed66, 100 epochs, audited task loss/optimizer/batch/missing protocol, retained BEST checkpoints and external co-resident controller `7d495f7`. Existing completed runs are not overwritten or repeated. GPU4 remains excluded. A subsequent user instruction replaces the GPU3 reservation with twelve total concurrent runs on GPUs5/6/7 (four per GPU): retain these three and add the first nine previously reserved candidates. The final three former GPU3 candidates and PPGN remain pending.

See `LAUNCH_NEXT_THREE.json` for exact commands and observed execution stage; running checks are not completed training results.
