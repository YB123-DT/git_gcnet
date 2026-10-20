# MOSEI seed66: original GCNet missing Flat vs one-stage Router

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

Two fresh60epoch trajectories, original MOSEI official train/dev/test.
Each rate selects its own TEST W-F1-best epoch; ACC uses the same predictions.
Fixed legal common GCNet masks; no extra training holdout or validation selection.
Eight-rate scores may combine eight checkpoints, not one deployable model.

| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 | Delta W-F1 pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 87.3693 | 87.4794 | 87.2653 | 87.4039 | +0.1386 |
| 0.1 | 86.0759 | 86.6263 | 85.9573 | 86.5323 | +0.5750 |
| 0.2 | 85.4155 | 85.6081 | 85.2688 | 85.4842 | +0.2154 |
| 0.3 | 84.7001 | 85.0578 | 84.5401 | 84.8881 | +0.3481 |
| 0.4 | 82.8288 | 82.6637 | 82.6560 | 82.3537 | -0.3023 |
| 0.5 | 82.0583 | 81.8932 | 81.7656 | 81.6447 | -0.1209 |
| 0.6 | 80.4073 | 79.8019 | 80.0390 | 79.5161 | -0.5229 |
| 0.7 | 79.0864 | 78.3709 | 78.7406 | 78.3442 | -0.3963 |
| mean | 83.4927 | 83.4377 | 83.2791 | 83.2709 | -0.0082 |
| high | 80.5173 | 80.0220 | 80.1817 | 79.8350 | -0.3467 |

Seed66 only; no automatic extra seeds or methods.
Older MOSEI scores used different missing masks/budgets and are not paired comparators.
No independent test-generalization claim. No new architecture or auxiliary loss.
