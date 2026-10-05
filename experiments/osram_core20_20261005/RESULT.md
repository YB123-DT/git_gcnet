# C01–C20

INTERNAL DIAGNOSTIC ONLY

Implementation status: 20 mechanisms and two necessary controls implemented.
Performance status: PARTIAL RESULTS AVAILABLE. Correctness is not a gain.

## Progress at 2026-10-05T03:37:01.331820+00:00

Completed16 configurations (14 methods +2 controls), all100epochs with verified
selected artifacts. Active: C06 78/100. Pending: C08, C09, C15, C16, C19.
No completed method exceeds original Flat mean8 or high-missing mean.
BEST-per-rate W-F1 percentages below; deltas are percentage points against
historical regression Flat seed66 (81.068095 /76.352251). C17 requires binary
control; C20 requires three-exit mean control. See PROGRESS_20261005.json.

| Configuration | mean8 | high | mean8 delta vs Flat |
| --- | ---: | ---: | ---: |
| C20 | 80.646 | 75.884 | -0.423 |
| C20-mean-control | 80.632 | 75.505 | -0.436 |
| C07 | 80.514 | 76.257 | -0.554 |
| C05 | 80.013 | 75.430 | -1.056 |
| C11 | 79.928 | 75.187 | -1.140 |
| C17-binary-control | 79.619 | 74.863 | -1.449 |
| C13 | 79.579 | 74.296 | -1.489 |
| C04 | 79.494 | 74.897 | -1.574 |
| C02 | 79.401 | 74.220 | -1.667 |
| C12 | 79.245 | 74.645 | -1.823 |
| C01 | 79.018 | 73.901 | -2.050 |
| C14 | 78.481 | 73.481 | -2.587 |
| C10 | 78.242 | 72.889 | -2.826 |
| C03 | 59.397 | 59.411 | -21.672 |
| C17 | 53.915 | 54.255 | -27.153 |
| C18 | 50.828 | 53.414 | -30.241 |

C20 versus matched mean-control: +0.014pp mean8, not a demonstrated robust gain.
Single seed, Test-oracle INTERNAL DIAGNOSTIC ONLY; no significance claim or
automatic multi-seed promotion. Low C03/C17/C18 scores are retained, not hidden
or presumed code failures without a separate diagnosis.

2026-10-05 launch: source00522bd deployed to biggpu; GPU6 only.
Eleven actual training processes confirmed by optimizer-step RESOURCE.json:
C01/C02/C03/C04/C05/C07/C10/C11/C12/C13/C14. Remaining9 methods and2 controls
queued; max11 concurrent, high-resource methods admitted separately. No extra
GPU smoke. Launch evidence/commands/PIDs in LAUNCH.json. This records startup,
not completed100-epoch runs. Live dispatcher/logs/checkpoints:
/data2/yb/remote_experiments/osram_core20_20261005/.

Starting implementation commit: 073984d; actual branch:
feature/osram-uniform-forced-text. Historical Flat training commit is not known
from its original provenance; do not claim 073984d trained the reference.
Reference protocol/config: REFERENCE_CONFIG.json, copied from biggpu
osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json.
Reference audited 8-rate mean W-F1=81.068095%, high=.5/.6/.7 mean=76.352251%.
Those historical numbers use per-rate Test-oracle selection, not validation.

## Boundaries

Memory contents/algorithm change for C01/C02/C04–C10; Value quantization changes
C03. C11/C15 change observed representations. C12/C17 replace the task head or
distribution. C13/C14/C16 replace the classification readout with their actual
inference mechanism. C18/C19/C20 change the optimization objective/update.
Source-required objectives are documented per-method; not all runs are
emotion-loss-only. No generic residual substitute is presented as a full method.

C17 uses binary task supervision and TRAIN exemplar projection, with fixed
warm/joint/push/head phases; compare its C17-binary-control before attributing
differences to prototypes. C20 compares C20-mean-control with the same single
Memory trajectory and Local/Base/Full supervised exits. Both differ from the
original one-exit Flat protocol, and are not three independent missing views.

Configuration files list every difference. Fixed auxiliary coefficients are
first-transfer choices, not source-paper optima or test-selected sweeps.
Parameter counts are actual CPU constructors at confirmed raw dimensions
512/1024/1024; initial trainable counts do not describe all C17 phase freezes.

## Verification

69 CPU checks passed (8.80 seconds) for all core methods, actual full-model
task backward, true trainer loops for C15/C17/C18/C20, original emotion loss
and complete recovery regressions. No new dependencies installed.

Independent review corrected C02 singleton reconstruction (zero key gradient),
C03 moving-Value reconstruction target, and C11 moving-latent likelihood
targets. C03/C11 now reconstruct original observed raw features. C06 training
uses activation checkpointing with identical solver outputs/outer gradients.
Default-off path matches pre-change commit 073984d outputs and RNG exactly.
No additional GPU smoke or resource-profile runs: resource peaks are recorded
inside actual training. GPU6 only; target up to11 concurrent light runs.

C05 original author repository unavailable; implemented against paper math,
not claimed author-code-equivalent. All explicit adaptations/source pins are
in docs/osram_core20_20261005/*.json and groupdro.md.

## Results pending

Report every method's per-rate W-F1, 8-rate/high means, actual peaks/time,
status/failure and controls after completion. Do not treat diagnostics as
emotion-shift detection, reliability guarantees or paper-ready novelty.
