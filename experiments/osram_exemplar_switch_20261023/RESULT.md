# MOSI66 exemplar frozen switch: completed

**INTERNAL TEST-ORACLE DIAGNOSTIC. No retraining or new selection.**

## Main result

The label-aware branch is not inactive: ON improves the same-weight OFF readout
by0.332 W-F1 percentage points on average. However, this does not produce an
improvement over original Flat (+0.006pp), and its high-missing direct effect
is−0.026pp. The feature-only arm has only+0.042pp direct mean effect and a
negative high-missing effect. Do not expand this version automatically.

| Variant | Scope | Original Flat W-F1 | Same checkpoint OFF | Same checkpoint ON | ON−OFF | ON−Flat |
|---|---|---:|---:|---:|---:|---:|
| Feature-only | Eight-rate mean | 79.249 | 78.602 | 78.644 | +0.042 | −0.605 |
| Feature-only | .5/.6/.7 mean | 73.410 | 73.426 | 73.165 | −0.262 | −0.245 |
| Label-aware | Eight-rate mean | 79.249 | 78.923 | 79.255 | +0.332 | +0.006 |
| Label-aware | .5/.6/.7 mean | 73.410 | 73.045 | 73.019 | −0.026 | −0.391 |

The label-aware arithmetic decomposition is
`ON−Flat = (OFF−Flat) + (ON−OFF) = −0.326 +0.332 = +0.006pp`.
This separates frozen readout outcomes, not training causes. OFF is part of a
jointly trained model and may be outside its training distribution. Original
Flat also has different selected epochs. We cannot conclude that training
damaged the backbone, that freezing it would help, or that a new LR would solve
this merely from the decomposition.

## Label-aware per rate

Corrections/harms are **OFF→ON**, not original Flat→ON. Every row uses the
original ON-selected checkpoint, unchanged classification head and exact mask.

| Missing | Selected epoch | OFF W-F1 | ON W-F1 | ON−OFF | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|
| .0 | 42 | 88.281 | 87.816 | −0.464 | 0 | 3 |
| .1 | 38 | 84.486 | 84.486 | 0.000 | 1 | 1 |
| .2 | 48 | 82.900 | 82.482 | −0.419 | 3 | 6 |
| .3 | 38 | 81.763 | 81.926 | +0.163 | 2 | 1 |
| .4 | 28 | 74.820 | 78.274 | +3.454 | 38 | 21 |
| .5 | 38 | 76.397 | 75.990 | −0.407 | 4 | 7 |
| .6 | 32 | 73.647 | 74.142 | +0.495 | 10 | 7 |
| .7 | 36 | 69.092 | 68.926 | −0.165 | 6 | 7 |

The positive mean is concentrated at.4. Excluding that rate as a descriptive
check, the other seven direct deltas average−0.114pp; this is not a new model
selection criterion. At.4 the ON result is still below original Flat79.305.
This supports a condition-specific contribution, not consistent improvement.

Across all eight rate-specific evaluations, label-aware OFF→ON corrects64
cases and harms53; high-missing corrects20 and harms21. These sums repeat the
same utterances under different masks, not unique independent targets.
Feature-only totals are47/48 and high-missing12/18.

## MSE and actual output changes

| Variant | Scope | Flat MSE | OFF MSE | ON MSE |
|---|---|---:|---:|---:|
| Feature-only | Eight-rate | 1.494741 | 1.749037 | 1.507411 |
| Feature-only | High-missing | 1.849947 | 2.129839 | 1.823484 |
| Label-aware | Eight-rate | 1.494741 | 1.813962 | 1.468161 |
| Label-aware | High-missing | 1.849947 | 2.270773 | 1.782892 |

The label-aware mean absolute ON/OFF prediction change is about0.49, so this
is not a numerically negligible branch. Large score changes and better MSE do
not necessarily yield more correct polarity decisions. No causal interpretation
is assigned to residual norms or label retrieval merely from this result.

Complete ACC/W-F1/MSE, both arms' per-rate outcomes, prediction changes and
residual/anchor norms: [METRICS.csv](completed/METRICS.csv),
[REPORT.json](completed/REPORT.json), [generated report](completed/RESULT.md).
The already-completed original training results are preserved separately in
[training_reference/RESULT.md](training_reference/RESULT.md); their near-flat
overall score is not the same comparison as ON versus OFF here.

## Verified execution

- Inference source `1df92392a88b2714a38316fcff5f11d2b8604404`;
  original training source `d6f6eb19d9ef68403dcfe23a18a7ef66712a9602`.
- Actual current model dependency hashes match those recorded during training.
  Protected TRAIN bank, data, split, seed and exact masks were verified.
- Two variants×eight retained BEST checkpoints; all686 targets/rate, of which
  656 have nonzero polarity labels. No test-label filtering before inference.
- Primary ON/OFF use one unchanged CHM/Local/Flat forward; OFF reuses captured
  skip/adapter outputs, the same LayerNorm and classifier. OFF still has Memory.
- Original selected ON prediction max error **0 for all16 checkpoints**.
- Independent existing `enabled=False` verification max error **0 for all16**.
- Every selected checkpoint was strict-loaded; epoch/trajectory and saved
  checkpoint/prediction SHA256 checked; weight digests unchanged, no parameter
  gradients, no optimizer, Torch CPU/CUDA RNG unchanged.
- Four new tests PASS locally and remotely. Independent sklearn re-evaluation
  of all16 raw prediction files, full unique target coverage and correction/
  harm count checks PASS. Compile and whitespace checks PASS.
- biggpu physicalGPU7, UUID`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`, original
  `/data2/yb/reproduction_workspace/envs/s0/bin/python`, Torch2.2.2+cu121,
  NumPy1.24.3/sklearn1.3.2; unhealthy GPU4 excluded.
- Entire audit53.03sec including setup/loading/verification. Per-arm peak
  allocated1110.05MiB; this is not a standalone performance benchmark.
- Remote `/data2/yb/remote_experiments/osram_exemplar_switch_20261023`:
  `source_1df9239`, `execution_v1`, `run_v1.log`, `tests_v1.log`.
  Original checkpoints/Mask/predictions remain unchanged. Full new NPZ files
  stay remote. Metadata and checkpoint/mask/artifact hashes are in
  [completed/SUMMARY.json](completed/SUMMARY.json).

## Decision

This rules out the blanket explanation that the branch never affects decisions.
It does **not** show a stable new net classification benefit. Keep these records,
but stop this version without HPO/more seeds or automatic retraining. In
particular the result does not justify continuing on the assumption that the
residual is too weak. Other seeds, training interventions and mechanisms of
the differences remain untested; no new failure hypothesis is declared proven.

Only independent audit files/tests/results were added; no model or training
implementation was modified. No unrelated jobs were stopped or edited.
