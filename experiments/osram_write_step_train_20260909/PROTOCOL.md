# Fixed write-step retraining and crossed evaluation

User-approved single-variable experiment. INTERNAL DIAGNOSTIC ONLY.
Eta=.6 was selected after viewing Test results; retraining on these same data
does not produce independent generalization evidence.

## Locked implementation

Normal differentiable OSRAM.block_write keeps its ridge solve unchanged and
multiplies the correction by fixed write_step. Default=1; new training=.6.
No new learned parameter, optimizer LR change, frozen trainable modules,
slot removal, fusion changes, completion, protection, or extra loss.
Old checkpoint configs without the field resolve to1. Inference reads the
saved value; a named evaluation-only override is recorded explicitly.

## Sequence and comparisons

1. Verify default path, gradients, causal/padding behavior, CLI/config restore.
2. Reuse completed eta1 References only after config/budget/source-path review.
3. Train IEMOCAPFour seeds66–70 from scratch with eta=.6, 100epochs, inherited
   official features, fold5, cyclic eight-rate training, Adam LR=.001/batch32.
4. Evaluate each new best.pt with its saved eta=.6 (C), and eta=1 (D).
5. Repeat the same five-seed protocol on CMUMOSI, inherited fold1 settings.
   No IEMOCAPSix or alternate eta training.

A=train1/test1; B=train1/test.6; C=train.6/test.6; D=train.6/test1.
A/B reuse paired frozen-checkpoint diagnostic records from the previous grid
and range folders; do not retrain1 unless provenance fails.
For C/D, one checkpoint per seed chosen under training eta=.6 by the same
eight-rate-mean Test-oracle policy as the Reference. D never reselects epoch.
The paired A/B/C/D table uses rates0/.1/.3/.5/.7 (FIVE-rate mean), matching the
existing frozen diagnostics. Normal training retains all eight-rate results;
report those separately, not as though B/D have eight-rate evaluations.
Compare C-A, C-B, B-A, and D-C by rate and paired seed. Do not impose an ordering.

Record E_old=historical err_decay excluding NO_HISTORY and E_new=current
observed post-write fit. Post-write changes affect future reads only.
Record masks and checkpoint/config hashes; never silently use mismatched masks.

## Reuse review

All ten Reference configs have 100 completed epochs, initial_backbone_checkpoint
null, eta implicitly1, full OSRAM, mean fusion, forward-only, no slot reuse,
same seeds and cyclic schedule. Reference snapshots:
osram_forward_only_iemocap_20260908/iemocap4/seed_{66..70};
osram_forward_only_mosi_20260908/seed_{66..70} under remote_experiments.
Since d27b58f, model/trainer only gained disabled slot-reuse plumbing before
this change; OSRAM also gained diagnostics disabled by default. Review default
path regression before accepting reuse. Loss, masks, optimizer logic unchanged.

## Compute and artifacts

Use available GPU0/1, three seeds on GPU0 and two on GPU1, not GPU4.
Datasets run sequentially, seeds within a dataset concurrently. Each task writes
its own fresh output directory; failure stops the next dataset rather than
silently changing configuration. No resume from old best.pt or budget reduction.
Only small code/config/result artifacts go to Git; checkpoints/raw stay remote
and provenance identifies them. No new dependencies.

## Completion checklist

- Code/tests and old checkpoint compatibility verified.
- Exact config delta is osram_write_step only (legacy missing defaults normalized).
- Ten new training runs and twenty C/D evaluation runs completed.
- Paired masks and A/B/C/D source identity verified; no new checkpoint selection.
- Per-seed/rate table, paired deltas, five/eight-rate distinction and diagnostics.
- Git diff check and push; label pending outputs honestly until runs finish.
