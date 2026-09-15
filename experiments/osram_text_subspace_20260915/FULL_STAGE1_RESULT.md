# Seed66 full Stage1 and matched target-space gradient control

## Scope and decision

Completed unchanged Stage1 for 100 epochs / 200 optimizer steps on GPU6.
Selected epoch **94** by minimum validation composite loss (1.9822848),
not by sentiment W-F1. No Test batches consumed. **Do not launch full Stage2
on the basis of this comparison:** the auxiliary gradient scales are not comparable,
and validation sample-specific prediction has not improved jointly across A/V/AV.
No loss weights, temperature, architecture, or production training code changed.

Artifacts: `FULL_STAGE1.json`, `target_space_control_seed66/TARGET_SPACE_CONTROL.json`,
and `target_space_control_seed66/stage1_per_epoch.csv` (600 rows: 100 epochs,
two splits, three source patterns). The control JSON records checkpoint provenance,
Teacher/R hashes, history hash, and exact shared parameter names.

## Stage1 dynamics

Validation snapshot; predictability is the average pattern SmoothL1. Checkpoint
selection uses the full sentiment + predictability + variance + covariance loss.

| Epoch | Target W-F1 % | Target erank | Predictability loss | A centered cosine | V centered cosine | AV centered cosine |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84.386 | 1.841 | .08490 | .0076 | .0073 | .0127 |
| 25 | 86.090 | 3.162 | .04820 | .0637 | -.0312 | -.0263 |
| 50 | 85.163 | 4.286 | .05353 | .0928 | -.0510 | -.0340 |
| **94 selected** | **84.751** | **5.512** | **.05426** | **.1367** | **-.0587** | **-.0128** |
| 100 | 83.785 | 5.786 | .05287 | .1389 | -.0591 | -.0094 |

At epoch94, training target W-F1 is 87.441%, erank 5.698;
training A/V/AV centered cosine is .1639/.2410/.2705. Thus training predictability
does not transfer consistently to validation, especially for V/AV. Lower regression
loss alone is insufficient evidence of sample-specific prediction.

Selected validation prediction audit:

| Source | Centered cosine | Real-minus-shuffle cosine | Retrieval % (chance .437%) | Prediction/target std | C(prediction) W-F1 % |
|---|---:|---:|---:|---:|---:|
| A | .13668 | .004760 | .873 | .0974 | 42.688 |
| V | -.05867 | -.004188 | .873 | .1815 | 43.753 |
| AV | -.01281 | -.000833 | .873 | .1924 | 44.426 |

Retrieval is just 2/229 per pattern and must not be interpreted as significant
improvement over chance. C(prediction) uses the Stage1 sentiment head, not a refitted
prediction-space probe. Rank increased relative to epoch1; do not call this
training-induced R collapse or claim full-rank/noncollapsed useful predictions.

## Exactly matched Stage2 gradient control

One fresh seed66 Student, one rate=.5 training batch (380 missing-Text targets),
**one forward and one dropout realization**. Both losses reuse identical reg/cl
prediction tensors and Teacher targets. Frozen R comes from selected epoch94.
No optimizer update; all model/R hashes unchanged; parameter `.grad` remains None.

Both modes use .1 JEPA = .05 regression + .05 symmetric InfoNCE, temperature .03.
Coordinates are the same 2,179,616 parameters in the common autograd support of
emotion and JEPA across ObservedSetEncoder and OSRAM.

| Metric | Full Text256 | Predictable subspace32 |
|---|---:|---:|
| Emotion loss | 2.649124 | 2.649124 |
| Regression loss | .448668 | .166862 |
| InfoNCE loss | 6.570113 | 12.007116 |
| Weighted regression gradient norm | .006032 | .009639 |
| Weighted NCE gradient norm | .611680 | 2.663169 |
| Weighted JEPA gradient norm | .611572 | 2.662854 |
| Emotion gradient norm | 6.442153 | 6.442153 |
| JEPA / emotion norm | 9.493% | 41.335% |
| NCE / regression norm | 101.40x | 276.29x |
| Cosine(emotion, weighted JEPA) | .006231 | .007184 |

The subspace auxiliary gradient is **4.3541x** the full-space gradient. Both are
nearly orthogonal to emotion on this batch, not strongly conflicting, but this
does not remove the magnitude confound. Both auxiliary objectives are NCE dominated.
These are initial raw gradients, not clipped/Adam updates or training-long averages.
Fresh Flat zero-initialization gives zero emotion gradient in memory-related OSRAM
parameters on this first forward; OSRAM-only cosine is undefined (None), not zero.

Differences reflect the entire target-space change: learned R Jacobian, dimension,
geometry, and loss reduction, not effective rank alone. No automatic gradient
balancing or loss adjustment is introduced. Full Stage2 remains unstarted.

## Reproduction and verification

Remote environment: `/data2/yb/reproduction_envs/s0/bin/python3.10`, code root
`/data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple` on `biggpu`.

```bash
python experiments/osram_text_subspace_20260915/run.py --stage stage1 --seed 66 \
  --output /data2/yb/remote_experiments/osram_text_subspace_20260915/stage1_seed66_full100
python experiments/osram_text_subspace_20260915/compare_target_spaces.py
```

Default comparison output requires a new directory to avoid overwriting evidence.
Stage1 fitting itself took 9.04 seconds after frozen feature extraction; this is
small R/Q/C training, not 100 epochs of OSRAM. Checkpoints remain remote, not in Git.
The identity-projection regression test verifies exact equality of both comparison
paths when R is identity. Independent read-only review found no blocking issue.
Fresh remote verification: **50 tests passed** in 6.26 seconds across the text-subspace,
dynamics, target-space control, Teacher information, transfer, alignment, and
task-direction suites; one existing PyG deprecation warning. `git diff --check`
and Python compilation passed.

Limitations: one seed, one Stage2 gradient batch, no Stage2 training or Test results.
The complete CSV allows inspection beyond selected snapshots; no alternate checkpoint
was selected from post-hoc pattern metrics.
