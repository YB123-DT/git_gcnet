# Evidence-centered hierarchical decision correction

INTERNAL DIAGNOSTIC ONLY

Status: all three seeds completed 100 epochs on biggpu GPU0. No improvement
over original Flat in this screening. Code commit: 2b00639.

## Fixed experiment

MOSI seeds 66/67/68, 100 epochs each. Reference configs/checkpoints:
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_{66,67,68}`.
Current pre-change implementation: 29a9d7c. Original random cyclic rates 0.0–0.7,
batch32, Adam lr0.001, weight decay0.00001, original masks, MSE task primitive and
per-rate BEST Test-oracle selection stay unchanged. The new option
`--osram-decision-correction` replaces the Flat decision path and changes task
supervision to the fixed mean of three exits. It does not enable the previous
Gap-increment Filter, Relation, gate, completion, JEPA or paired-view training.

## Architecture

One unchanged causal Memory scan supplies Local and Base/Gap. Use causal forward
512-dimensional contexts, not their zero backward halves. Three separate MLPs
use LayerNorm(input) -> Linear(input,128) -> GELU -> Linear(128,C), no dropout.
H_L is normally initialized; D_B/D_G output weights zero-initialized and output
bias absent because a shared final bias would cancel in the difference.

```
sL   = H_L(L)
dB   = D_B(L,B) - D_B(L,0)
sLB  = sL + dB
dG   = D_G(L,B,masked_G,a) - D_G(L,B,0,a)
sLBG = sLB + dG
task = (original_task(sL,y) + original_task(sLB,y) + original_task(sLBG,y))/3
```

Both evaluations of a correction share weights and retain gradients. Padding,
inactive Gap and empty history are hard masked. Active parameters train jointly
in one stage. Old Flat/head must not execute in this path. Inference uses sLBG;
there is no label-dependent routing. Initialization recovers the NEW Local-only
head, not original Flat. Centering is not a guarantee of useful corrections.

## Execution

Server: biggpu. GPU4 prohibited. Prefer GPU0 for same-card parallel seeds,
resource permitting. All output directories/logs/snapshots are independent of
the completed Gap Filter experiment. Launch only after correctness tests and
specification/quality reviews pass. Final source/config/environment/GPU/reference
hashes go in PROVENANCE.json and measured active/retained counts in PARAMETERS.json.

Runner command from the isolated snapshot:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u experiments/osram_decision_correction_20261003/run.py --output /data2/yb/remote_experiments/osram_decision_correction_20261003/seed_66 --seed 66 --gpu 0 --commit CODE_COMMIT
```

Repeat for 67/68 with their own output paths. This is from-scratch training,
not checkpoint resume. Eight BEST checkpoints and original mask hashes are
required per seed; 100-epoch history and successful provenance are required
before reporting completion. Main scores use Full output's per-rate BEST.
Local/LB exit metrics at the same Full-selected checkpoint are diagnostics,
not independently selected oracle outputs.

## Baseline summary

| Seed | Original Flat 8-rate W-F1 (%) | High (.5/.6/.7) |
|---|---:|---:|
| 66 | 81.068 | 76.352 |
| 67 | 80.556 | 75.990 |
| 68 | 80.053 | 74.440 |
| Mean | 80.559 | 75.594 |

Completed results are below. No additional configuration sweep is authorized.

## Measured capacity and pre-launch checks

| Model | Stored total parameters | Trainable | Retained inactive |
|---|---:|---:|---:|
| Original Flat | 13,509,793 | 13,509,793 | 0 |
| Decision correction | 13,943,592 | 3,992,487 | 9,951,105 |

The new decision head contains 433,799 parameters. Old Flat/readout parameters
remain in the state dict for compatibility but are never executed and are
excluded from the optimizer. This is NOT a parameter-matched comparison; the
active decision architecture is substantially smaller than original Flat.

Default-off tests compare model AND OSRAM directly with pre-change source
29a9d7c, including RNG, common parameter initialization and nonzero Flat outputs.
Standalone tests verify learned independent B=0/G=0 cancellation in train/eval,
C=1/C=6, fixed conditioners, differentiable paired calls, safe masked NaNs,
single scan, unchanged Memory tensors and strict checkpoint reload. Training
tests verify equal task losses and original evaluation thresholds/valid labels.

Actual cfg84-dimension synthetic CUDA smoke on biggpu V100: two optimizer steps,
one subsequent evaluation, finite parameters/gradients and exact first-valid
Local=LB=Full. Synthetic batch=4, sequence length=10; peak allocated CUDA memory
164.2 MiB. This is a correctness smoke, not a full-batch memory capacity claim
or a model performance result. See cuda_smoke.json and verification logs.

Fresh verification: 29 local tests and 33 remote V100 tests passed; scoped
compileall and git diff --check passed. Remote archive excludes three Git-object
legacy tests that passed locally. Specification and code-quality reviews passed
without blocking findings. Six core/runner source SHA256 hashes matched local
files before launch. GPU0 was checked idle with 32,495 MiB free.

## Launch record

Remote root: `/data2/yb/remote_experiments/osram_decision_correction_20261003`.
Immutable source snapshot: `code/`. GPU0 UUID:
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45` (process-visible cuda:0).

| Seed | PID | Log | Output |
|---|---:|---|---|
| 66 | 3995805 | seed_66.log | seed_66/ |
| 67 | 4050144 | seed_67.log | seed_67/ |
| 68 | 4104634 | seed_68.log | seed_68/ |

Started 2026-10-03 at 13:26 UTC, staggered by first-epoch verification and fresh
GPU identity/free-memory checks. Independent persistent processes run concurrently;
this is not serial training. Logs/checkpoints are separate. See launch.json for
exact commands/timestamps. Earlier SSH transport resets occurred before formal
launch; no training process was duplicated or migrated to a local GPU.

## Completed three-seed results

W-F1 (%); deltas are percentage points. Arithmetic mean across the eight rates
within each seed, then equal-weight across seeds. High missing = .5/.6/.7.
All scores use per-rate BEST Test-oracle; INTERNAL DIAGNOSTIC ONLY.

| Seed | Flat 8-rate | Decision 8-rate | Delta | Flat high | Decision high | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 66 | 81.068 | 79.575 | -1.493 | 76.352 | 74.781 | -1.572 |
| 67 | 80.556 | 78.892 | -1.664 | 75.990 | 74.176 | -1.814 |
| 68 | 80.053 | 79.403 | -0.650 | 74.440 | 73.537 | -0.903 |
| Mean | 80.559 | 79.290 | -1.269 | 75.594 | 74.164 | -1.429 |

Sample SD across seeds (ddof=1): Flat/Decision eight-rate 0.507/0.355;
high missing 1.016/0.622. No significance claim from this exploratory three-seed
screening and no claim that centering alone caused the loss. This experiment
changes readout structure, active capacity and supervision simultaneously.

| Rate | Flat mean | Decision mean | Delta | Decision seed66 | seed67 | seed68 |
|---|---:|---:|---:|---:|---:|---:|
| .0 | 88.419 | 87.252 | -1.167 | 86.988 | 87.153 | 87.616 |
| .1 | 85.845 | 84.551 | -1.294 | 84.715 | 84.324 | 84.615 |
| .2 | 83.431 | 81.695 | -1.737 | 81.010 | 82.743 | 81.330 |
| .3 | 80.999 | 80.405 | -0.595 | 80.054 | 79.238 | 81.922 |
| .4 | 78.996 | 77.923 | -1.073 | 79.489 | 75.149 | 79.131 |
| .5 | 77.327 | 75.795 | -1.531 | 75.746 | 73.803 | 77.836 |
| .6 | 75.848 | 74.108 | -1.741 | 73.816 | 73.896 | 74.611 |
| .7 | 73.607 | 72.590 | -1.016 | 74.780 | 74.828 | 68.162 |

Verification: each provenance status is complete; each history contains exactly
100 epochs; all eight BEST checkpoints exist; each seed's evaluation mask
hashes match its original Flat reference; provenance configs differ only by
the enabled decision flag; recorded source hashes match the immutable snapshot.
Full exit metrics exactly equal the main reported scores. Original run PIDs
have exited. Archived results/ contains model metrics/config/provenance/parameter
counts plus corresponding baseline metrics/config. No new inference or training
was performed to produce this report.

## Three exits at the same Full-selected checkpoints

These are cumulative outputs of ONE jointly trained model, not separately
trained Local/Base ablations and not independently selected exit checkpoints.

| Scope | Seed | Local | Local+Base | Full |
|---|---:|---:|---:|---:|
| 8-rate | 66 | 75.6783 | 79.4109 | 79.5749 |
| 8-rate | 67 | 77.2317 | 78.7534 | 78.8919 |
| 8-rate | 68 | 76.9709 | 78.9949 | 79.4029 |
| 8-rate | Mean | 76.6270 | 79.0531 | 79.2899 |
| High | 66 | 70.3764 | 74.7466 | 74.7807 |
| High | 67 | 71.5727 | 74.0574 | 74.1759 |
| High | 68 | 70.3251 | 73.4485 | 73.5367 |
| High | Mean | 70.7581 | 74.0842 | 74.1644 |

Sample SD (Local/LB/Full): 8-rate 0.8318/0.3326/0.3552;
high 0.7060/0.6495/0.6221. Local->LB gains +2.4261 points overall,
+3.3261 high; LB->Full gains +0.2368 overall, +0.0803 high.

| Scope | Transition | Corrections | Harms | Repeated evaluation exposures |
|---|---|---:|---:|---:|
| 8-rate | L->LB | 755 | 350 | 15,744 |
| 8-rate | LB->Full | 130 | 89 | 15,744 |
| High | L->LB | 453 | 238 | 5,904 |
| High | LB->Full | 59 | 52 | 5,904 |

Counts pool seeds/rates and are NOT unique utterances or independent samples.
Each seed/rate evaluates 686 valid utterances, with 656 nonzero-label utterances
used for binary W-F1/transitions. Threshold remains prediction>0 and label>0.
At rate0 all seeds have exact zero Gap correction and identical LB/Full metrics,
with zero corrections/harms. Gap is not uniformly helpful across rates: at .6,
mean LB74.5149 -> Full74.1079 (-0.4071 points).

Interpretation: Base corrections provide positive aggregate incremental value
within the new architecture, and Gap adds a smaller aggregate increment. Thus
failure to beat original Flat does not mean the correction branches learned
nothing. However, these conditional exit gains do not establish superiority
over Flat or identify whether readout capacity, objective weighting or centered
parameterization caused the overall deficit. Retain original Flat as baseline;
do not automatically expand this configuration.
