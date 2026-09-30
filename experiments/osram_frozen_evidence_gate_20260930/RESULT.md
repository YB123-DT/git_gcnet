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

## Best comparison by missing rate

Three-seed means; differences computed before rounding, in percentage points.
Original parent scores versus Stage2 best including epoch0, same Test-oracle scope.

| Miss | Original Flat best | Frozen Flat + Gate best | Delta pp |
|---|---:|---:|---:|
| 0.0 | 88.419 | 88.421 | +0.003 |
| 0.1 | 85.845 | 86.003 | +0.157 |
| 0.2 | 83.431 | 83.761 | +0.330 |
| 0.3 | 80.999 | 81.127 | +0.128 |
| 0.4 | 78.996 | 79.272 | +0.276 |
| 0.5 | 77.327 | 77.535 | +0.209 |
| 0.6 | 75.848 | 75.883 | +0.034 |
| 0.7 | 73.607 | 73.928 | +0.322 |
