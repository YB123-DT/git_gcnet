# Seed-direction check and one proposed follow-up

INTERNAL TEST-ORACLE DIAGNOSTIC

Source: experiments/osram_mosi_source_role_direct_3seed_20261020/completed/METRICS.csv
SHA256: d5e8f75a0499c3d629149045e853eac006c97d5cd3c16b4e4ceb6a125b2565ad

No new training or inference was performed. This note only analyzes already
completed three-seed results and inspects the existing source-audit scope.

| Rate | Seed66 Delta W-F1 pp | Seed67 | Seed68 |
|---|---:|---:|---:|
| 0.0 | 0.411 | -0.218 | 0.325 |
| 0.1 | 0.219 | -0.387 | -0.477 |
| 0.2 | 0.919 | -0.300 | 0.223 |
| 0.3 | -0.792 | 0.131 | -0.110 |
| 0.4 | 1.397 | 0.410 | -0.546 |
| 0.5 | 0.913 | 0.048 | -0.905 |
| 0.6 | 0.116 | 0.051 | -1.280 |
| 0.7 | -0.610 | -1.114 | 0.192 |

All8rates contain at least one positive and one negative seed delta. This
confirms heterogeneous *total trained-model performance*, not the sign of any
particular Value-source contribution. Near-zero deltas may be metric noise;
no statistical significance or mechanistic cause is inferred from this table.

The existing scalar12-source audit explicitly covers original Flat seed66
only,using its fixed miss=.3 checkpoint for all conditions. It does not cover
the new direct-source checkpoints or original Flat seeds67/68. Therefore it
cannot establish that the three seeds have conflicting12-cell causal roles.
Source-role histories record last-training-batch averaged gamma,not complete
fixed-cohort test-time per-source removal effects. Gamma itself is a scale,
not a signed task-utility measurement.

## One suggested NEXT ACTION: paired neutral-weight readout audit

NOT executed by this note. No new module,optimizer or training is proposed.
Use each seed/rate's already selected direct checkpoint and its exact masks.
Compute original CHM/Local/source reads once,then replay only the existing
readout with learned gamma,all active gamma=1,and12individual active slots
restored to1. Inactive Gap/source/padding slots remain inactive; never add
unobserved sources or alter Memory/query/write trajectories.

For each role/source slot i,define Delta_i=W-F1(learned)-W-F1(gamma_i=1),
on the identical valid active cohort. Positive means its learned adjustment
helped under this intervention; negative means it hurt. Also retain ACC,
corrections/harms and N. Cells without valid evidence are N/A. Different row
cohorts or12nonlinear marginal effects must not be summed into overall W-F1.

All-gamma=1 remains the newly trained checkpoint,NOT the original Flat. Compare:
new_learned minus original_Flat = (new_neutral minus original_Flat) +
(new_learned minus new_neutral),for any one scalar metric on a fixed cohort.
This separates training-associated baseline-path change from the direct
contribution of learned weighting,without claiming either is a pure semantic
causal effect. Neutral source sums reproduce raw reads only within numerical
tolerance; verify reconstruction and prediction consistency explicitly.

Three seed-specific4x3tables would test whether the learned calibration's
benefit/harm directions differ. They are not the same question as removing
the source entirely(gamma=0),which tests reliance on that evidence. Do not
substitute one intervention for the other or infer success from gate means.

This targeted test may show useful learned weights offset by backbone change,
harmful weights,or weak weights with little direct effect. It is meant to
distinguish these outcomes,not preselect a diagnosis or justify another module.
