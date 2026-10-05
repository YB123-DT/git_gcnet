# Four source-grounded transfers

**INTERNAL DIAGNOSTIC ONLY** — per-rate Test-oracle checkpoint selection.
No formal paper claim and no hyperparameter sweep.

Integration baseline commit: `2c361c9`. Baseline: cfg84 causal no-JEPA Flat,
MOSI seed66, 100 epochs, cyclic .0–.7;
Adam/lr/batch/task loss/evaluation masks inherited from the exact seed66
reference config. Baseline 8-rate W-F1 81.068095%, high-missing 76.352251%.

| Method | Changed mechanism | Fixed transfer choice |
|---|---|---|
| R02 DeltaProduct | Historical storage update | Two ordered, noncommuting delta writes; observed-only pooled K/V; beta in (0,2); ungated; read-before-write |
| R03 Mesa | Historical storage and read | Cumulative G/H ridge statistics; exact solve, not CG; positive diagonal ridge init1 floor.25; original alpha/beta initial values; write_step .6 scales statistic input |
| R12 RNC | Training objective | Single-view joint MSE + .1 RNC; temperature2; non-normalized Euclidean distance; up to128 uniformly sampled valid utterances per batch |
| R18 LUPI | Training-only task-hidden noise | Complete TRAIN raw current-utterance features condition mean1 Gaussian noise; MSE + .001 unsquared sigma norm / sqrt(valid activations); eval identity |

R02/R03 intentionally replace the original block-write algorithm, not a readout
residual. R12's single-stage joint training and R18's coefficient/norm scaling
are explicit task adaptations, not exact reproduction of original schedules.
All preserve ObservedSetEncoder, Base/Gap query interface, Flat and task head.
R02 does not use the original alpha/beta or .6 proximal write-step: its learned
internal beta performs the two new delta steps. Those original gates remain
loadable but are frozen. R03 clones alpha/beta into its new statistic controller;
the original gate parameters are frozen. Legacy OSRAM gate diagnostics describe
the retained original parameters, not R02's internal beta.
R18 never routes complete features into encoder, query, keys/values or memory.
No Gate, completion, JEPA, paired views, persistent mix or additional sweep.

Source PDFs, source pins and licensing notes:
`docs/osram_r4_source_audit_20261005/`. R12 is independently implemented from
the equation because the inspected author repository has no license.

| Method | Added parameters | Total trainable parameters |
|---|---:|---:|
| Flat reference | 0 | 13,509,793 |
| R02 | 33,024 | 13,542,785 |
| R03 | 544 | 13,510,305 |
| R12 | 0 | 13,509,793 |
| R18 | 534,208 | 14,044,001 |

R02/R03 freeze32 original controller parameters, so added parameters differ
from net trainable increase. Counts use actual raw dims512/1024/1024, output1600.

## Verification and execution

Focused tests: module equations, causal read-before-write, padding/inactive
observation safety, tie-preserving ranking, finite gradients and updates,
privileged-free evaluation; real-model backward and one R18 training step.
On biggpu's existing s0 environment with CUDA hidden: **21 tests passed**;
one existing torch-geometric deprecation warning. `git diff --check` passed.
Local integration test collection was blocked by missing torch-geometric;
no dependency was installed and the real experiment environment was used.
Execution status and source commit are recorded in `LAUNCH.json` when submitted.
Each run saves PROVENANCE.json, config.json, RESOURCE.json, train.log,
last_training.pt, eight BEST checkpoints and predictions. No result is claimed
before these artifacts complete.

Server: biggpu, GPU6 UUID `GPU-e4cafb17-818e-216a-b94a-7440063a9153` only.
Queue: `python -m experiments.osram_r4_20261005.dispatch --root RUN_ROOT
--reference FLAT_SEED66_CONFIG --data-manifest DATA_MANIFEST`.
Per-method runner: `python -m experiments.osram_core20_20261005.run --method
R02 --reference FLAT_SEED66_CONFIG --data-manifest DATA_MANIFEST --output
METHOD_OUTPUT --gpu-uuid GPU-e4cafb17-818e-216a-b94a-7440063a9153`.
The same command applies to R03/R12/R18. No automatic multi-seed expansion.
