# MOSEI Coupled Source Role: interim seed66 progress

INTERNAL TEST-ORACLE SCREEN

Checked 2026-10-20T13:37:29.046713+00:00; 47/60epochs,process2381857,biggpu GPU7.
Mode: residual(.25*tanh shared source-role correction), not direct2*sigmoid.
MOSEI direct-source weighting and additional MOSEI seeds have NOT been started.
Existing MOSI direct three-seed runs remain separate and untouched.

| Rate | Flat ACC | Current ACC | Flat W-F1 | Current W-F1 | Delta W-F1 pp | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.3693 | 87.4518 | 87.2653 | 87.4304 | +0.1651 | 22 |
| 0.1 | 86.0759 | 86.1585 | 85.9573 | 86.0897 | +0.1324 | 22 |
| 0.2 | 85.4155 | 85.2504 | 85.2688 | 85.3029 | +0.0341 | 18 |
| 0.3 | 84.7001 | 84.5625 | 84.5401 | 84.5752 | +0.0351 | 18 |
| 0.4 | 82.8288 | 82.7188 | 82.6560 | 82.4779 | -0.1781 | 30 |
| 0.5 | 82.0583 | 81.7006 | 81.7656 | 81.6317 | -0.1339 | 27 |
| 0.6 | 80.4073 | 80.4073 | 80.0390 | 79.9258 | -0.1132 | 28 |
| 0.7 | 79.0864 | 78.9488 | 78.7406 | 78.3305 | -0.4101 | 30 |
| mean | 83.4927 | 83.3998 | 83.2791 | 83.2205 | -0.0586 | — |
| high | 80.5173 | 80.3522 | 80.1817 | 79.9627 | -0.2191 | — |

Current results are per-rate best-so-far through47epochs. Flat completed60;
its same-budget47prefix happens to contain the same selected best scores.
Residual method currently slightly below Flat, not evidence of final improvement
or statistical inferiority. The remaining13epochs are already authorized;
no restarts,parameter changes or extra experiments were performed for this check.

Checks: same original initialization,only requested source-role config differs,
same split/seed/test banks. All47completed epochs passed actual training-mask/
update-count/LR parity;14connected/14nonzero new parameter tensors last epoch.
Original LR remains.001 throughout60epochs. One-stage scalar MSE,GCNet random
missing,original Flat1600/Local Skip/CHM maintained. No task Router.

Selected artifact metric recomputation and complete-only report remain pending
the final60epochs. Current values are existing selected-epoch summary metadata.
Source f6bc52e5e4f83ff13be642595cfa4775ed996e01.
