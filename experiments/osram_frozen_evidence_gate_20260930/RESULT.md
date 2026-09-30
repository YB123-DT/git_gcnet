# Frozen Flat / Gate-only Stage2 results

INTERNAL TEST-ORACLE DIAGNOSTIC, NOT VALIDATION-SELECTED PAPER RESULTS.

All24 tasks (3 seeds ×8 parent checkpoints) completed100 epochs, exit codes0.
All2400 recorded epoch frozen-state checks passed. Original parameters/buffers
remained unchanged; only Gate was trained. Parent weights are not overwritten.

| W-F1 %, mean ± seed sample SD | Original Flat / epoch0 | Oracle best including epoch0 | Final epoch100 |
|---|---:|---:|---:|
| Eight-rate mean |80.559 ±0.507|80.741 ±0.491|80.512 ±0.552|
| High missing .5/.6/.7 |75.594 ±1.016|75.782 ±0.890|75.414 ±0.961|
| Eight-rate delta pp |—|+0.182|−0.047|
| High-missing delta pp |—|+0.188|−0.180|

| Seed | Flat eight-rate | Best | Final | Best delta | Final delta |
|---|---:|---:|---:|---:|---:|
|66|81.068|81.212|81.016|+0.144|−0.052|
|67|80.556|80.779|80.598|+0.223|+0.042|
|68|80.053|80.233|79.922|+0.180|−0.131|

Five tasks select epoch0;19 select a trained epoch with a strict improvement.
Because epoch0 is a best candidate, nonnegative oracle-best deltas are guaranteed
by selection. The small best gain does not establish deployable improvement.
Final epochs do not improve overall/high-missing averages, so this experiment
does not demonstrate a stable Gate benefit after freezing Flat. No significance
or causal claim that joint adaptation was the sole prior failure cause is made.

Summary recomputed from archived seed means; all seeds/rates retained without
outlier exclusions. Separate final and selected prediction scores are verified
from existing NPZ files (original nonzero-label filter, threshold >0, weighted F1).
No new inference or training was performed during result aggregation.

Source code88937c7, launch recorda239fed; detailed results in results/SUMMARY.json
and per-task metrics/provenance. Each task retains best.pt and last.pt remotely;
Gate-only checkpoints require the immutable referenced parent Flat checkpoint.
